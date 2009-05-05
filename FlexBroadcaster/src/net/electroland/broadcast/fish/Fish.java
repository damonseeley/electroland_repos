package net.electroland.broadcast.fish;

import java.text.DecimalFormat;
import java.text.NumberFormat;

import net.electroland.broadcast.server.XMLSocketMessage;

/**
 * A Fish is a representation of (you guessed it) a fish in the pool that the
 * Flash clients are playing.  Each fish has (x,y) coordinates, depth, 
 * orientation, speed, an obejctId, an objectType, a state, and a current frame.
 *  
 * @author geilfuss
 */
public class Fish implements Comparable<Fish>, XMLSocketMessage{

	public static final int NOSTATE = -91919;
	
	// for efficiency, we're not using getters or setters.  just access these directly.
	public int objectType, frame, depth, state, accent;
	public double scale, x, y, orientation, speed;
	public String objectId, movieFileName;
//	public int seq = 0;
	private NumberFormat f = new DecimalFormat("#.#");
	private NumberFormat fTwoDec = new DecimalFormat("#.##");

	public Fish(String objectId, int objectType,
				double x, double y, double scale,
				double orientation, 
				int depth, double speed, 
				int state, int accent, int frame, String movieFileName){

		this.objectType = objectType;
		this.objectId = objectId;

		this.scale = scale;
		this.x = x;
		this.y = y;
		this.orientation = orientation;
		this.depth = depth;
		this.speed = speed;

		this.state = state;
		this.accent = accent;
		this.frame = frame;
		this.movieFileName = movieFileName;
	}

	/* (non-Javadoc)
	 * @see net.electroland.fish.messaging.XMLSocketMessage#toXML()
	 */
	public String toXML() {
		
		StringBuffer b = new StringBuffer();
		// very simple attribute names to keep the message size small.
		b.append("<fish");
		b.append(" id=\"").append(objectId).append('"');
		b.append(" t=\"").append(objectType).append('"');
		b.append(" x=\"").append(f.format(x)).append('"');
		b.append(" y=\"").append(f.format(y)).append('"');
		b.append(" v=\"").append(f.format(speed)).append('"');
		b.append(" d=\"").append(f.format(depth)).append('"');
		b.append(" o=\"").append(f.format(orientation)).append('"');
		b.append(" s=\"").append(f.format(state)).append('"');
		b.append(" a=\"").append(accent).append('"');
		b.append(" p=\"").append(fTwoDec.format(scale)).append('"'); // think 'p' for percent.
		b.append(" f=\"").append(frame).append('"');
//		b.append(" q=\"").append(seq).append("\"");
		if (movieFileName != null){
			b.append(" m=\"").append(movieFileName).append('"');			
			movieFileName = null;
		}
		b.append("/>");
		
		if(state == -1) {
			state = NOSTATE;
		}
		accent = NOSTATE;
		
		
		
		return b.toString();
		
		
	}

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	public int compareTo(Fish o) {
		// sort by z-index.
		return depth;
	}	
}