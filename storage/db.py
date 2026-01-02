from io import BytesIO
import sqlite3
from typing import Dict, Optional
from PIL import Image
import os
from utils.logger import get_logger

logger = get_logger()

class ImageDB:
    def __init__(self, db_path: str = "storage/images.db"):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, ".."))
        db_path = os.path.join(project_root, "storage", "images.db")

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT UNIQUE NOT NULL,
                    name TEXT,
                    data BLOB NOT NULL
                )
            """)

    def store_image(
        self,
        image: Image.Image,
        face_id,
        name: Optional[str] = None
    ):
        face_id = str(face_id)
        with BytesIO() as output:
            image.save(output, format="PNG")
            img_bytes = output.getvalue()

        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO images (face_id, name, data)
                VALUES (?, ?, ?)
            """, (face_id, name, img_bytes))
        logger.info(f"Image stored in DB with face_id: {face_id}")

    def retrieve_by_face_id(self, face_id) -> Optional[Dict]:
        face_id = str(face_id)
        logger.info(f"Attempting to retrieve face_id: {face_id}")
        cur = self.conn.execute(
            "SELECT data, name FROM images WHERE face_id=?",
            (face_id,)
        )
        row = cur.fetchone()

        if not row:
            logger.info(f"No image found for face_id: {face_id}")
            return None

        img_bytes, name = row
        logger.info(
            f"Image found for face_id: {face_id}, "
            f"data size: {len(img_bytes)} bytes, "
            f"name: {name}"
        )


        return face_id, name
    
    def save_image_to_path(self, face_id, output_path: str):
        face_id = str(face_id)
        logger.info(f"Saving image for face_id: {face_id} to {output_path}")
        image = self.retrieve_by_face_id(face_id)
        if image is not None:  # Explicit None check instead of falsy check
            try:
                image.save(output_path)
                logger.info(f"Image successfully saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving image to {output_path}: {e}")
        else:
            logger.info(f"No image retrieved for face_id: {face_id}, cannot save")
    def get_unnamed_faces(self):
        cur = self.conn.execute(
            "SELECT id, face_id, name FROM images WHERE name IS NULL OR name = ''"
        )
        rows = cur.fetchall()
        unnamed_faces = [{"face_id": row[1], "name": row[2]} for row in rows]
        logger.info(f"Retrieved {len(unnamed_faces)} unnamed faces")
        return unnamed_faces
    
    def get_image_by_face_id(self, face_id) -> Optional[Image.Image]:
        face_id = str(face_id)
        cur = self.conn.execute(
            "SELECT data FROM images WHERE face_id=?",
            (face_id,)
        )
        row = cur.fetchone()

        if not row:
            logger.info(f"No image found for face_id: {face_id}")
            return None

        img_bytes = row[0]
        image = Image.open(BytesIO(img_bytes))
        logger.info(f"Image retrieved for face_id: {face_id}")
        return image
    
    def update_name(self, face_id, name: str):
        face_id = str(face_id)
        with self.conn:
            self.conn.execute(
                "UPDATE images SET name=? WHERE face_id=?",
                (name, face_id)
            )
        logger.info(f"Updated name for face_id: {face_id} to {name}")

    def close(self):
        self.conn.close()