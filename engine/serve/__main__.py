"""Run the CPU-only echo server."""

import uvicorn

from .api import create_stub_app


def main() -> None:
    uvicorn.run(create_stub_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
