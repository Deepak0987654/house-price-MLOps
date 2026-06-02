from pydantic import BaseModel, Field, field_validator
from app.core.config import CURRENT_YEAR

# pydantic model to validate incoming data
class UserInput(BaseModel):

    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality of the house (1 = very poor, 10 = excellent).")
    GrLivArea: int = Field(..., gt=0, description="Above-ground living area in square feet (excluding basement).")
    GarageCars: int = Field(..., ge=0, le=3, description="Number of cars that can fit in the garage.")
    TotalBsmtSF: int = Field(..., ge=0, description="Total basement area in square feet.")
    YearBuilt: int = Field(..., ge=1800, le=CURRENT_YEAR, description="Year the house was originally built.")
    YearRemodAdd: int = Field(..., ge=1800, le=CURRENT_YEAR, description="Year of the most recent remodeling or renovation.")
    FullBath: int = Field(..., ge=0, le=3, description="Number of full bathrooms (with shower or bathtub).")

    KitchenQual: str = Field(..., description="Kitchen quality rating (Ex = Excellent, Gd = Good, TA = Typical/Average, Fa = Fair, NA = Not Available).")
    ExterQual: str = Field(..., description="Exterior material quality rating (Ex = Excellent, Gd = Good, TA = Typical/Average, Fa = Fair, NA = Not Available).")
    BsmtQual: str = Field(..., description="Basement height/quality rating (Ex = Excellent, Gd = Good, TA = Typical/Average, Fa = Fair, NA = Not Available).")

    GarageArea: int = Field(..., ge=0, description="Size of the garage in square feet.")
    LotArea: int = Field(..., gt=0, description="Total lot size in square feet.")

    # Normalize categorical values
    @field_validator("KitchenQual", "ExterQual", "BsmtQual")
    @classmethod
    def validate_quality(cls, v):
        allowed = ["EX", "GD", "TA", "FA", "NA"]
        v = v.upper()
        if v not in allowed:
            raise ValueError("choose between EX, GD, TA, FA, NA")
        return v

    # 🔥 Cross-field validation
    @field_validator("YearRemodAdd")
    @classmethod
    def check_years(cls, v, info):
        year_built = info.data.get("YearBuilt")
        if year_built and v < year_built:
            raise ValueError("YearRemodAdd must be >= YearBuilt")
        return v