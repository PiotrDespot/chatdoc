from typing import Annotated
from fastapi import Depends
from app.config import Settings, get_settings
from sqlalchemy.ext.asyncio import AsyncSession
from app.postgres.db import get_session


AppSettings = Annotated[Settings, Depends(get_settings)]
DatabaseSession = Annotated[AsyncSession, Depends(get_session)]
