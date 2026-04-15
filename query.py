import sqlite3

conn = sqlite3.connect('./data/imdb.db')
agent1_conn = sqlite3.connect('./data/agent1.db')
agent2_conn = sqlite3.connect('./data/agent2.db')
agent3_conn = sqlite3.connect('./data/agent3.db')
agent4_conn = sqlite3.connect('./data/agent4.db')
agent5_conn = sqlite3.connect('./data/agent5.db')
# people = agent1_conn.execute("SELECT * FROM people").fetchall()
# crew = agent2_conn.execute("SELECT * FROM crew").fetchall()
# ratings = agent3_conn.execute("SELECT * FROM ratings").fetchall()
# episodes = agent3_conn.execute("SELECT * FROM episodes").fetchall()
# titles_half = agent4_conn.execute("SELECT * FROM titles").fetchall()
# titles_half_2 = agent5_conn.execute("SELECT * FROM titles").fetchall()
# akas = agent5_conn.execute("SELECT * FROM akas").fetchall()

# Find movies named Casablanca and their ratings
# SELECT t.title_id, t.type, t.primary_title, t.premiered, t.genres, r.rating, r.votes
# FROM titles t  INNER JOIN ratings r ON ( r.title_id = t.title_id  )
# WHERE t.primary_title = 'Casablanca' AND t.type = 'movie'
# [
#     ('tt0034583', 'movie', 'Casablanca', 1942, 'Drama,Romance,War', 8.5, 654433),
#     ('tt0135830', 'movie', 'Casablanca', 1961, 'Documentary', 5.2, 17),
#     ('tt10399586', 'movie', 'Casablanca', 2023, 'Romance', 5.0, 18)
# ]
# movies = conn.execute("""
# SELECT t.title_id, t.type, t.primary_title, t.premiered, t.genres, r.rating, r.votes
# FROM titles t
# INNER JOIN ratings r ON ( r.title_id = t.title_id  )
# WHERE t.primary_title = 'Casablanca' AND t.type = 'movie'""").fetchall()
# print(movies)


# people = agent1_conn.execute("SELECT name, died FROM people WHERE name LIKE '%Brigitte%'").fetchall()
# # people = agent1_conn.execute("SELECT * FROM people LIMIT 5").fetchall()
# print(people)

query = "SELECT person_id, crew.* FROM crew WHERE title_id='tt2015381' AND characters LIKE '%Rocket%' -- AND category='voice' "
result = conn.execute(query).fetchall()
print(result)