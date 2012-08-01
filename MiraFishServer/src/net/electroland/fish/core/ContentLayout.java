package net.electroland.fish.core;

import javax.vecmath.Vector3f;

import net.electroland.fish.boids.VideoFish;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.Content;
import net.electroland.fish.util.FishProps;

public class ContentLayout {
	
	public static final float videoDepth =  FishProps.THE_FISH_PROPS.getProperty("videoDepth", 1.75f);
	public float edgeBuffer = FishProps.THE_FISH_PROPS.getProperty("videoEdgeBuffer", 25f);
	
	public static enum Edge  { top, left, bottom, right };
	
	
	Bounds bounds;
	Slot[] leftSlots;
	Slot[] rightSlots;
	
	Slot[] topSlots;
	Slot[] bottomSlots;
	
	public void reset() {
		leftSlots[0].vid = null;
		leftSlots[1].vid = null;
		rightSlots[0].vid = null;
		rightSlots[1].vid = null;
		topSlots[0].vid = null;
		topSlots[1].vid = null;
		topSlots[2].vid = null;
		bottomSlots[0].vid = null;
		bottomSlots[1].vid = null;
		bottomSlots[2].vid = null;
	}
	
	// by increasing x/y values
	
	public ContentLayout(Bounds worldBounds) {
		bounds = worldBounds;
		
		leftSlots = new Slot[2];
		rightSlots = new Slot[2];
		
		topSlots = new Slot[3];
		bottomSlots = new Slot[3];
		
		
		leftSlots[0] = new Slot(384, Edge.left);
		leftSlots[1] = new Slot(1152, Edge.left);

		rightSlots[0] = new Slot(384, Edge.right);
		rightSlots[1] = new Slot(1152, Edge.right);
		
		
		topSlots[0] = new Slot(690, Edge.top);
		topSlots[1] = new Slot(1536, Edge.top);
		topSlots[2] = new Slot(2382, Edge.top);

		bottomSlots[0] = new Slot(690, Edge.bottom);
		bottomSlots[1] = new Slot(1536, Edge.bottom);
		bottomSlots[2] = new Slot(2382, Edge.bottom);

		
	}
	
	
	public Slot getFreeSlot(Vector3f pos) { 
		Slot s = getSlot(pos);
		if(s.vid == null) return s;
		return null;
	}
	
	
	
	
	public Slot getSlot(Vector3f pos) {
		if(pos.y < bounds.getCenterY()) { // top half of screen
			if(pos.x < 1024) { //left 3rd
				if(pos.x < pos.y) { // left edge
					return leftSlots[0];
				} else { // top edge
					return topSlots[0];
				}
			} else if(pos.x < 2045) { //middle third
				return topSlots[1];				
			} else { // right third
				if(bounds.getRight()-pos.x < pos.y ) { // right edge
					return rightSlots[0];
				} else { // top
					return topSlots[2];
				}
			}
		} else { // bottom half
			if(pos.x < 1024) { //left 3rd
				if(pos.x < bounds.getBottom() - pos.y) { // left edge
					return leftSlots[1];
				} else { // top edge
					return bottomSlots[0];
				}
			} else if(pos.x < 2045) { //middle third
				return bottomSlots[1];				
			} else { // right third
				if(bounds.getRight()-pos.x < bounds.getBottom() - pos.y ) { // right edge
					return rightSlots[1];
				} else { // top
					return bottomSlots[2];
				}
			}
		}
	}
	
	
	public class Slot {
		float dist; // dist from 0 on edge
		public VideoFish vid;
		Edge edge;
		
		
		public Slot(float dist, Edge e) {
			this.dist = dist;
			edge = e;
		}

		public void getMediaLocation(Content c, Vector3f destPos, Vector3f destHeading) {
			destPos.z = videoDepth;
			destHeading.set(0,0,0);
			switch(edge) {
			case left:
				destPos.y = dist;
				destPos.x = c.halfHeight + edgeBuffer;
				destHeading.y = 1;
				break;
			case right:
				destPos.y = dist;
				destPos.x = bounds.getRight() - c.halfHeight - edgeBuffer;
				destHeading.y = -1;
				break;
			case top:
				destPos.x = dist;
				destPos.y = c.halfHeight + edgeBuffer;
				destHeading.x = -1;
				break;
			case bottom:
				destPos.x = dist;
				destPos.y = bounds.getBottom() - c.halfHeight - edgeBuffer;
				destHeading.x = 1;
				break;
			}
			
		}

		
	}

}
