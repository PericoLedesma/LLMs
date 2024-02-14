import asyncio
import time

async def say_after(delay, what):
    print(f'starts {what}')
    await asyncio.sleep(delay)
    print(what)

def main():
    print(f"started at {time.strftime('%X')}")

    asyncio.run(say_after(3, 'hello'))
    print('middle')
    asyncio.run(say_after(2, 'world'))

    print(f"finished at {time.strftime('%X')}")

main()