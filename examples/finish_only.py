"""Negative-control agent: immediately finish without reading answers or folding."""
class Agent:
    def request_public(self, text, image_paths):
        return dict(action={'op':'finish'},input_tokens=0,output_tokens=0,raw_output='finish-only negative control')

def create():
    return Agent()
