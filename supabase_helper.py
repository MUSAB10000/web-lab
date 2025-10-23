from supabase import create_client, Client
import os
import bcrypt
from dotenv import load_dotenv
import datetime
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Hash password
def hash_password(plain_password: str) -> str:
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    return hashed.decode()

# ✅ Login (check email + password)
def login_admin(email: str, password: str):
    response = supabase.table("users").select("*").eq("email", email).execute()
    if not response.data:
        print("❌ No user found with that email.")
        return None

    user = response.data[0]
    db_password = user["password"]

    try:
        # ✅ Case 1: password is already hashed (bcrypt)
        if bcrypt.checkpw(password.encode("utf-8"), db_password.encode("utf-8")):
            print("✅ Login successful (hashed password).")
            return user["id"]
    except ValueError:
        # ✅ Case 2: plain-text password (not hashed)
        if password == db_password:
            print("⚠️ Plain-text password detected — hashing and fixing it now...")
            hashed_pw = bcrypt.hashpw(db_password.encode("utf-8"), bcrypt.gensalt()).decode()
            supabase.table("users").update({"password": hashed_pw}).eq("id", user["id"]).execute()
            print("✅ Password converted to hashed bcrypt format.")
            return user["id"]

    print("❌ Invalid password.")
    return None

def save_video(title, path, user_id):
    response = supabase.table("videos").insert({
        "title": title,
        "video_name": path,
        "uploaded_by": user_id
    }).execute()

    if not response or response.data is None:
        print("❌ Failed to save video. Check Supabase table columns or connection.")
        return None

    print(f"✅ Video saved with ID: {response.data[0]['id']}")
    return response


def save_person(
    video_id, track_id, frame_number,
    has_mask, has_gloves, has_labcoat, has_glasses,
    in_red_zone, status="unsafe"
):
    response = supabase.table("persons").insert({
        "video_id": video_id,
        "track_id": track_id,
        "frame_number": frame_number,
        "has_mask": has_mask,
        "has_gloves": has_gloves,
        "has_labcoat": has_labcoat,
        "has_glasses": has_glasses,
        "in_red_zone": in_red_zone,
        "status": status
    }).execute()
    return response


# ✅ Save detection
def save_detection(person_id, class_name, confidence, bbox_dict, frame_path):
    return supabase.table("detections").insert({
        "person_id": person_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox_dict,
        "frame_path": frame_path
    }).execute()

# ✅ Save alert
def save_alert(person_id, alert_type, reason):
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    
    return supabase.table("alerts").insert({
        "person_id": person_id,
        "alert_type": alert_type,
        "reason": reason,
        "created_at": timestamp
    }).execute()


# ✅ Save clip
def save_clip(person_id, alert_id, clip_path, start_frame, end_frame):
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    
    return supabase.table("clips").insert({
        "person_id": person_id,
        "alert_id": alert_id,
        "clip_path": clip_path,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "created_at": timestamp
    }).execute()
