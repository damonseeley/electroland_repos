package net.electroland.fish.util;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.vecmath.Vector2f;
import javax.vecmath.Vector3f;

import net.electroland.fish.ui.Drawable;


public class Bounds implements Drawable {
	private float top;
	private float left;
	private float right;
	private float bottom;

	private float near; // bigger
	private float far; // far

	
	private float centerX;
	private float centerY;

	private float width;
	private float height;
	private float depth;

	public static enum OUTER_REGION { inside, left, right, topLeft, top, 
		topRight, bottomLeft, bottom, bottomRight };
		public Bounds() {

		}

		public Bounds(Bounds b) {
			this(b.top, b.left, b.bottom, b.right, b.near, b.far);
		}

		public Bounds(float top, float left, float bottom, float right, float near, float far) {
			this.top = top;
			this.left = left;
			this.right = right;
			this.bottom = bottom;
			this.near = near;
			this.far = far;

			width = right - left;
			height = bottom - top;
			depth = near-far;
			
			
			centerX = left + (width * .5f);
			centerY = top + (height * .5f);
			
			//TODO remove this check
			if(near < far) {
				System.out.println("swap near and far");
			}
		}
		
		public String toString() {
			return top + "," + left + "," + bottom + "," + right + "," + near + "," + far;
		}
		
		public static Bounds valueOf(String s) {
			String[] b = s.split(",");
			return new Bounds(
					Float.valueOf(b[0]),
					Float.valueOf(b[1]),
					Float.valueOf(b[2]),
					Float.valueOf(b[3]),
					Float.valueOf(b[4]),
					Float.valueOf(b[5])
					);
		}

		public float getTop() {
			return top;
		}

		public void setTop(float top) {
			this.top = top;
			height = bottom - top;
			centerY = top + (height * .5f);
		}

		public float getLeft() {
			return left;
		}


		public void setLeft(float left) {
			this.left = left;
			width = right - left;
			centerX = left + (width * .5f);
		}

		public float getRight() {
			return right;
		}


		public void setRight(float right) {
			this.right = right;
			width = right - left;
			centerX = left + (width * .5f);
		}

		public float getBottom() {
			return bottom;
		}

		public void setBottom(float bottom) {
			this.bottom = bottom;
			height = bottom - top;
			centerY = top + (height * .5f);
		}

		public float getNear() {
			return near;
		}

		public void setNear(float near) {
			this.near = near;
			depth = near-far;
		}

		public float getFar() {
			return far;
		}

		public void setFar(float far) {
			this.far = far;
			depth = near-far;
		}

		public float getWidth() {
			return width;
		}

		public float getHeight() {
			return height;
		}
		public float getDepth() {
			return depth;
		}
		
//		public String toString() {
//			return "net.electroland.fish.util.Bounds[top="+top+ ",left="+left+ ",bottom="+bottom+",right="+right+ ",near="+near+ ",far="+far+"]";
//		}

		public boolean contains(float x, float y) {
			return (x >= left) && (x <= right) && (y >= top) && (y <= bottom);
		}

		public boolean contains(float x, float y, float z) {
			return (x >= left) && (x <= right) && (y >= top) && (y <= bottom) && (z <= near) && (z >= far);
		}
		public boolean contains(Vector3f vec) {
			return contains(vec.x, vec.y, vec.z);
		}

		public boolean contains(Vector2f vec) {
			return contains(vec.x, vec.y);
		}

		public void draw(GLAutoDrawable drawable, GL gl) {
			gl.glRectf(left, top, right, bottom);		
		}

		public OUTER_REGION getRegion(float x, float y) {
			if(x <= left) {
				if(y <= top) {
					return OUTER_REGION.topLeft;
				} else if(y >= bottom) {
					return OUTER_REGION.bottomLeft;				
				} else {
					return OUTER_REGION.left;
				}	
			} else if(x >= right) {
				if(y <= top) {
					return OUTER_REGION.topRight;
				} else if(y >= bottom) {
					return OUTER_REGION.bottomRight;				
				} else {
					return OUTER_REGION.right;
				}	
			} else { // between left and right
				if(y <= top) {
					return OUTER_REGION.top;
				} else if (y >= bottom) {
					return OUTER_REGION.bottom;
				} else {
					return OUTER_REGION.inside;
				}
			}
		}	

		public OUTER_REGION getRegion(Vector3f position) {
			return getRegion(position.x, position.y);
		}

		public float getCenterX() {
			return centerX;
		}

		public float getCenterY() {
			return centerY;
		}
}
