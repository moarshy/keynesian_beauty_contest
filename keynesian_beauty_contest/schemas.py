from pydantic import BaseModel

class InputSchema(BaseModel):
    num_agents: int
    batch_size: int = 5
