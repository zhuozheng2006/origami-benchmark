"""One whole-document policy for every model, no private steps or automatic crop."""
from pathlib import Path
import json
from PIL import Image
VERSION='document-autonomy-v12-dev2'
GUIDE='''You control a finite-thickness paper simulator to follow an origami document. Image 1 is a document page (or YOUR requested crop); image 2 is the ACTUAL current paper with a fixed board coordinate scale. Other pages are available. Read the task scope and numbered diagram instructions yourself. The system does NOT identify the next instruction or tell you when the work is complete.
Return ONE JSON action, no prose. Use the actual TOP view to place a straight crease in CURRENT WORLD/BOARD coordinates: x right, y up, 15cm equals 1 unit; an unrotated 15cm square has corners (0,0),(1,1); the supplied sheet may instead be a rectangle or hexagon, as shown; origin of rotations at board (0.5,0.5). A rotated paper can extend beyond [0,1], allowed [-0.5,1.5]. Diagram orientation may differ: explicitly rotate the paper when needed; NEVER renormalize the folded outline.
Actions:
{"op":"fold_line","line":[[x1,y1],[x2,y2]],"moving_point":[x,y],"layers":"all","sense":"toward","angle_deg":180}
line is the CREASE, NOT the arrow target. moving_point must lie on the moving flap. layers is all (every layer on that side), top (uppermost connected panel under the point), or bottom (lowest panel). toward lifts toward +z/the camera; away folds behind. Valley arrows normally use toward; mountain use away. Angle may be less than 180. Read arrow tail to head. Only straight creases already represented in the frozen mesh can execute; the current mesh hinge is fitted within a fixed 0.6mm geometric instrument band; no answer-aware snapping or measurement tool.
{"op":"unfold"} physically opens your latest retained fold in forward time, preserving plastic material history. This is NOT undo. Paper turns update the stored fold axis. Folds must be unfolded in reverse application order.
{"op":"turn_over","axis":"y"} physically flips the entire paper 180 degrees around world x or y. {"op":"rotate_paper","degrees":90} physically rotates in the plane, positive counterclockwise.
{"op":"view_page","page":1,"crop":[0,0,1,1]} selects a 1-based page and optional normalized image rectangle (x right,y DOWN). This only changes document view; it does not change paper or advance any instruction.
{"op":"undo"} reverts your last accepted PAPER operation; it does NOT refund attempts, cost, or history. Document view changes do not enter the undo stack.
{"op":"finish"} ends whenever YOU judge the requested scope complete. Acceptance only means a motion executed; it never certifies diagram correctness. There is no automatic final-review hint.
Failed calls, navigation, undo and finish all consume attempts. All public attempt history follows. A failure leaves the paper unchanged only when rollback_verified is true. Do not repeat an unchanged rejected request without a reason. You may recover or change strategy. No private target/expected-step count is visible.'''
def prompt(obs):
 return GUIDE+'\nPUBLIC STATE: '+json.dumps(dict(task=obs['task'],document=obs['document'],paper_state_hash=obs['current']['state_hash'],retained_paper_actions=obs['accepted_actions'],last_result=obs['previous_action_result'],history=obs['attempt_history'],budget=obs['attempt_budget']),ensure_ascii=False)
def prepare_inputs(obs,output,turn):
 path=Path(output)/f'document-view-{turn:03}.png';im=Image.open(obs['diagram']).convert('RGB');w,h=im.size;c=obs['document']['crop'];im.crop((round(c[0]*w),round(c[1]*h),round(c[2]*w),round(c[3]*h))).save(path)
 return [str(path),obs['current_image']]
def prepare_action(action,obs):
 if not isinstance(action,dict) or action.get('op') not in ('fold_line','unfold','turn_over','rotate_paper','view_page','undo','finish'):raise ValueError('Unknown public action')
 return action
