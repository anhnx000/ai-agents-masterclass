import asyncio

async def add(x, y):
    print('Fast function executed', x+y)
    return x + y

async def add_slow(x, y):
    await asyncio.sleep(1)
    print('Slow function executed', x+y)
    return x + y

async def main():
    asyncio.create_task(add_slow(4, 2), name='add_slow')
    asyncio.create_task(add(1, 2), name='add_fast')
    # Wait for all tasks to complete, run them concurrently
    await asyncio.gather(*asyncio.all_tasks())
    print('Done')
    

if __name__ == '__main__':
    asyncio.run(main())
    
