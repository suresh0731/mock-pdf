from pydantic import BaseModel


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h
