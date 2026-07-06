from fastapi import APIRouter

api_router = APIRouter(prefix="/api", tags=["api 路由"])


@api_router.get("/hello", summary="hello 接口", description="你好")
async def hello_world():
    return {"message": "Hello World"}
