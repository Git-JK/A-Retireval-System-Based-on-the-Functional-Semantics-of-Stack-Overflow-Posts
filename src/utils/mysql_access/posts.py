import pymysql
from utils.text_process.puretext_extract import puretext_extract
from ..config import Mysql_addr, Mysql_user, Mysql_password, Mysql_dbname_sotorrent

class DBPosts:
    db = pymysql.connect(host=Mysql_addr, user=Mysql_user, password=Mysql_password, database=Mysql_dbname_sotorrent)
    cursor = db.cursor()
    
    def collect_posts_all(self):
        answer_cursor = self.db.cursor()
        sql = f"select `Id`, `Body`, `Tags`, `Title`, `Score`, `ViewCount`, `FavoriteCount`, `AcceptedAnswerId` from `Posts` where `PostTypeId` = 1 and `AnswerCount` > 0"
        try:
            self.cursor.execute(sql)
            while True:
                row  = self.cursor.fetchone()
                if row is None:
                    break
                item = {
                    'Id': row[0],
                    'Body': row[1],
                    'Tags': row[2],
                    'Title': row[3],
                    'Score': row[4],
                    'ViewCount': row[5],
                    'FavoriteCount': row[6],
                    'PostLink': 'stackoverflow.com/questions/' + str(row[0])
                }
                answer_sql = f"select `Id`, `Body`, `Score` from `Posts` where `ParentId` = {row[0]}"
                answer_cursor.execute(answer_sql)
                answer_rows = answer_cursor.fetchall()
                answers = [{
                    'Id': r[0],
                    'Body': r[1],
                    'Score': r[2],
                    'Accepted': bool(r[0] == row[7])
                } for r in answer_rows]
                item['Answers'] = answers
                yield item
        except Exception as e:
            print(e)
    def collect_trainset_posts(self):
        answer_cursor = self.db.cursor()
        sql = f'select `Id`, `Body`, `Tags`, `Title`, `Score`, `ViewCount`, `FavoriteCount`, `AcceptedAnswerId` from `Posts` where `PostTypeId` = 1 and `AcceptedAnswerId` is not NULL and `Score` > 7 and `DeletionDate` is NULL and `ClosedDate` is NULL'
        try:
            self.cursor.execute(sql)
            while True:
                row  = self.cursor.fetchone()
                if row is None:
                    break
                item = {
                    'Id': row[0],
                    'Body': row[1],
                    'Tags': row[2],
                    'Title': row[3],
                    'Score': row[4],
                    'ViewCount': row[5],
                    'FavoriteCount': row[6],
                    'PostLink': 'stackoverflow.com/questions/' + str(row[0])
                }
                answer_sql = f"select `Id`, `Body`, `Score` from `Posts` where `ParentId` = {row[0]}"
                answer_cursor.execute(answer_sql)
                answer_rows = answer_cursor.fetchall()
                answers = [{
                    'Id': r[0],
                    'Body': r[1],
                    'Score': r[2],
                    'Accepted': bool(r[0] == row[7])
                } for r in answer_rows]
                item['Answers'] = answers
                yield item
        except Exception as e:
            print(e)
    def collect_search_range_posts(self):
        answer_cursor = self.db.cursor()
        sql = f'select `Id`, `Body`, `Tags`, `Title`, `Score`, `ViewCount`, `FavoriteCount`, `AcceptedAnswerId` from `Posts` where `PostTypeId` = 1 and `AcceptedAnswerId` is not NULL and `DeletionDate` is NULL and `ClosedDate` is NULL'
        try:
            self.cursor.execute(sql)
            while True:
                row  = self.cursor.fetchone()
                if row is None:
                    break
                item = {
                    'Id': row[0],
                    'Body': row[1],
                    'Tags': row[2],
                    'Title': row[3],
                    'Score': row[4],
                    'ViewCount': row[5],
                    'FavoriteCount': row[6],
                    'PostLink': 'stackoverflow.com/questions/' + str(row[0])
                }
                answer_sql = f"select `Id`, `Body`, `Score` from `Posts` where `ParentId` = {row[0]}"
                answer_cursor.execute(answer_sql)
                answer_rows = answer_cursor.fetchall()
                answers = [{
                    'Id': r[0],
                    'Body': r[1],
                    'Score': r[2],
                    'Accepted': bool(r[0] == row[7])
                } for r in answer_rows]
                item['Answers'] = answers
                for answer in answers:
                    if answer['Accepted'] == 1:
                        item['AcceptedAnswer'] = answer
                        break
                yield item
        except Exception as e:
            print(e)