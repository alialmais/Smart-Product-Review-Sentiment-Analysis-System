from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

STAFF_PASSWORD = "alialmais"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.password != STAFF_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    return credentials.username  # Return username if authentication is successful
