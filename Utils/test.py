import psycopg2 as psy

async def call():
    async with await psy.AsyncConnection.connect("dbname=test user=postgres") as aconn:
        async with aconn.cursor() as acur:
            await acur.execute("INSERT INTO test (num, data) VALUES (%s, %s)",(100, "abc'def"))
            await acur.execute("SELECT * FROM test")
            await acur.fetchone()
            # will return (1, 100, "abc'def")
            async for record in acur:
                print(record)
                
# Guardar cierto frame del video
# cv2.imwrite("./Frames/4m.png", frame)