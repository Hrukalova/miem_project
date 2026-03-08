import asyncio
from sqlalchemy import text
from core.db import async_session

async def check_status():
    async with async_session() as session:
        res = await session.execute(text("SELECT status, count(*) FROM documents GROUP BY status"))
        print("\n📊 Статусы документов в базе:")
        for row in res:
            print(f"- {row[0]}: {row[1]} шт.")

        # Заодно вернем failed обратно в очередь, если они есть
        await session.execute(text("UPDATE documents SET status = 'pending' WHERE status = 'failed'"))
        await session.commit()
        print("\n✅ Все файлы со статусом 'failed' снова переведены в 'pending'!")

asyncio.run(check_status())
