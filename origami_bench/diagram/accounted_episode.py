"""One execution entry for inference, validation, actions and accounting.

Legacy Episode remains frozen for reproducibility; its paper/physics semantics
are reused. Only this coordinator may submit actions in new evaluation runs.
"""
import time, copy
from .attempt_ledger import AttemptLedger, atomic_json
from .episode import Episode

def rejection(error):
    message=str(error);prefix=message.split(':',1)[0]
    category=prefix if prefix in ('model_abstention','phase_error','duplicate_action') else 'invalid_action_geometry' if any(x in message for x in ('straddles','Constructed crease','Coincident','moving_point')) else 'model_protocol_error'
    return dict(status='rejected',category=category,detail=message,rollback_verified=True)

class AccountedEpisode:
    def __init__(self,episode,output_cap_per_call=384):
        self.ep=episode;self.ledger=AttemptLedger(episode.output/'attempts.sqlite3',episode.max_decisions,output_cap_per_call)
    def _record_rejected(self,action,result,before):
        self.ep.events.append(dict(index=len(self.ep.events),instruction_index=len(self.ep.actions),action=action,before=before,after=self.ep.backend.state()['hash'],low_level_actions=[],result=result))
        atomic_json(self.ep.output/'events.json',self.ep.events)
    def turn(self,agent,policy):
        i=self.ledger.begin();start=time.monotonic();before=None;response={};event_count=len(self.ep.events)
        try:
            before=self.ep.backend.state()['hash'];obs=self.ep.observation()
            obs.update(attempt_history=self.ledger.public_history(),attempt_budget=self.ledger.budget())
            atomic_json(self.ep.output/f'observation-{i:03d}.json',obs)
            images=policy.prepare_inputs(obs,self.ep.output,i);prompt=policy.prompt(obs)
            # The request itself is persisted before external inference starts.
            atomic_json(self.ep.output/f'request-{i:03d}.json',dict(prompt=prompt,images=images,attempt_id=i))
            response=agent.request_public(prompt,images);response['public_image_files']=images
            self.ledger.response(i,response)
            atomic_json(self.ep.output/f'model-turn-{i:03d}.json',response)
            try:fold=policy.prepare_action(response.get('action'),obs)
            except (ValueError,TypeError,KeyError) as ex:
                result=rejection(ex);self._record_rejected(response.get('action'),result,before)
            else:
                atomic_json(self.ep.output/f'adapter-turn-{i:03d}.json',dict(model_action=response['action'],controller_action=fold,converter=policy.VERSION))
                result=self.ep.step(fold)
            event=copy.deepcopy(self.ep.events[-1]);event.update(episode_finished=self.ep.finished,wall_seconds=time.monotonic()-start)
        except Exception as ex:
            result=dict(status='rejected',category='environment_error',detail=type(ex).__name__+': inference/observation/execution failed; outcome requires audit',rollback_verified=False)
            if before is not None and len(self.ep.events)==event_count:
                try:result['rollback_verified']=self.ep.backend.state()['hash']==before;self._record_rejected(response.get('action'),result,before)
                except Exception:pass
            event=dict(action=response.get('action'),before=before,after=self.ep.backend.state()['hash'] if before is not None else None,result=result,episode_finished=False,wall_seconds=time.monotonic()-start,private_exception=str(ex))
        self.ledger.finish(i,event);self.ledger.export()
        stop=self.ep.finished or result.get('category') in ('environment_error','model_abstention')
        return result,stop
    def finish(self,agent_kind):
        report=self.ep.finish(agent_kind);value=self.ledger.export()
        report['attempt_budget']=value['budget'];report['attempt_ledger_version']='sqlite-v1'
        atomic_json(self.ep.output/'report.json',report);return report
    def close(self):self.ep.close();self.ledger.close()
