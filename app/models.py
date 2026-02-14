from typing import Optional
from sqlmodel import Field, SQLModel

class UserProfile(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True) 
    skin_tone: str
    undertone: str
    face_shape: str
    summary: str