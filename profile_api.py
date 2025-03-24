import cProfile
import pstats
import io
import uvicorn
from api import app  # Import FastAPI app

def profile_server():
    """Profiles FastAPI server performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)  # Show top 20 slowest functions
    
    with open("api_profile.txt", "w") as f:
        f.write(stream.getvalue())

if __name__ == "__main__":
    profile_server()
