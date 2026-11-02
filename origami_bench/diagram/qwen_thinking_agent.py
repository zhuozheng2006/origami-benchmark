"""SSH transport to an isolated open-weight visual policy; no model API fees."""
import subprocess,json,base64,selectors,time,os
from pathlib import Path


def public_prompt(observation):
    # Do not send IDs, source-group labels, answer files or run-directory names.
    task=observation['task']
    return json.dumps(dict(instruction=task['instruction'],paper_frame=task['paper_frame'],action_guide=observation['action_guide'],accepted_actions=observation['accepted_actions'],previous_action_result=observation['previous_action_result'],decisions_used=observation['decisions_used'],max_decisions=observation['max_decisions'],images=['First: instruction diagram, may contain several numbered panels.','Second: current paper top and oblique views, not a desired state.'],response_contract='Return one fold action or {"op":"finish"} after all diagram steps are executed. Read the dashed/dash-dot crease, arrow moving side and front/back. Coordinates are MATERIAL coordinates given in paper_frame. You may undo. Do not infer the answer from the schema example.'),ensure_ascii=False)


class QwenThinkingAgent:
    def __init__(self,log_path,numbered_images=False):
        self.log=Path(log_path).open('w')
        command='cd /data/ckpt/zhengzhuo/origami-benchmark && /data/ckpt/zhengzhuo/miniconda3/envs/origami-vlm/bin/python scripts/run_bounded.py --cpus 4 --group 6 /data/ckpt/zhengzhuo/miniconda3/envs/origami-vlm/bin/python scripts/serve_qwen32_thinking.py'
        if numbered_images: command+=' --numbered-images'
        self.proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ServerAliveInterval=30','4090B',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0)
        self.buffer=b'';self.sel=selectors.DefaultSelector();self.sel.register(self.proc.stdout,selectors.EVENT_READ)
        ready=self.read(900)
        if not ready.get('ok'):self.close();raise RuntimeError(ready.get('error'))
        self.metadata=ready['metadata']
    def read(self,timeout=900):
        deadline=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            if not self.sel.select(max(0,deadline-time.monotonic())):raise TimeoutError('VLM worker timed out')
            chunk=os.read(self.proc.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('VLM worker connection closed')
            self.buffer+=chunk
        line,self.buffer=self.buffer.split(b'\n',1);return json.loads(line)
    def act(self,observation):
        return self.request_public(public_prompt(observation),[observation[k] for k in ('diagram','current_image')])
    def request_public(self,text,image_paths):
        request=dict(text=text,images=[base64.b64encode(Path(p).read_bytes()).decode() for p in image_paths])
        self.proc.stdin.write((json.dumps(request)+'\n').encode());self.proc.stdin.flush()
        response=self.read()
        if not response.get('ok'):raise RuntimeError(response.get('error'))
        response['public_prompt']=request['text']
        return response
    def close(self):
        if self.proc.poll() is None:
            try:self.proc.stdin.write(b'{"op":"close"}\n');self.proc.stdin.flush();self.proc.wait(timeout=10)
            except Exception:self.proc.terminate()
        self.sel.close();self.log.close()
