package net.electroland.fish.core;

import java.awt.Point;
import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.vecmath.Vector2f;
import javax.vecmath.Vector3f;

import net.electroland.fish.ui.Drawable;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.FishProps;
import net.electroland.presenceGrid.net.GridDetectorDataListener;





public class SpacialGrid implements Drawable, GridDetectorDataListener {
	public Cell[][] cells;


	float xScale;
	float yScale;

	float cellWidth;
	float cellHeight;
	
	int gridWidth;
	int gridHeight;

	Bounds bounds;
	


	public SpacialGrid(Bounds bounds, int xCellCnt, int yCellCnt) {
		this.bounds = bounds;
		
		xScale =  (float) xCellCnt / (float)bounds.getWidth() ;
		yScale =  (float) yCellCnt / (float)bounds.getHeight() ;
		
		cellWidth = (int) Math.ceil((float)bounds.getWidth() / (float) xCellCnt);
		cellHeight = (int) Math.ceil((float)bounds.getHeight() / (float) yCellCnt);
		
		gridWidth  = xCellCnt;
		gridHeight = yCellCnt;
		
		cells = new Cell[gridWidth][gridHeight];

		
		for(int x = 0; x < gridWidth; x++) {
			for(int y = 0; y < gridHeight; y++) {
				cells[x][y] = new Cell(x,y);

			}

		}
		
		String str =FishProps.THE_FISH_PROPS.getProperty("cams", "0,0,65,45,1492");
		String[] strA = str.split(",");
		int i = 0;
		while(i < strA.length) {
			int xOffset = Integer.parseInt(strA[i++]);
			int yOffset = Integer.parseInt(strA[i++]);
			int w = Integer.parseInt(strA[i++]);
			int h = Integer.parseInt(strA[i++]);
			int port = Integer.parseInt(strA[i++]);
			new VisionListener(this, port, w, h, xOffset, yOffset);
		}

	
		System.out.println("SpacialGrid created");
	}


	public Vector<Boid> getBoidsInRadius(Vector3f pos, float radius) {

		float radiusSqr = radius * radius;

		Point topLeft = getGridLocation(pos.x - radius, pos.y-radius);
		Point botRight = getGridLocation(pos.x + radius, pos.y+radius);

		Vector<Boid> boids = new Vector<Boid>();

		for(int x = topLeft.x ; x <= botRight.x; x++) {
			for(int y = topLeft.y ; y <= botRight.y; y++) {
				// this can be optimized to automatically include all boids in cells contained in radius (and exclude those outside) 
				// but I don't think its worth the optimization
				Cell cell = cells[x][y];
				Enumeration<Boid> e = cell.boids.elements();
				while(e.hasMoreElements()) {
					Boid b = e.nextElement();
					float dx = b.position.x - pos.x;
					float dy = b.position.y - pos.y;

					if(radiusSqr >= ((dx * dx) + (dy * dy))) { 
						boids.add(b);
					}

				}
			}			
		}

		return boids;
	}

	public Point getGridLocation(float x, float y) {

		x = x - bounds.getLeft();
		y = y - bounds.getTop();

		x = (x < bounds.getRight()) ? x : bounds.getRight()-1;
		y = (y < bounds.getBottom()) ? y : bounds.getBottom()-1;

		x =  (x > 0) ? x : 0;
		y =  (y > 0) ? y : 0;






		return new Point((int) (x * xScale), (int) (y * yScale)); 	

	}

	public Point getGridLocation(Vector3f pos) {
		return getGridLocation(pos.x, pos.y);
	}


	public Point move(Point oldGridLoc, Vector3f newPosition, Boid boid) {
		remove(oldGridLoc, boid);
		Point newLoc = getGridLocation(newPosition);
		add(newLoc, boid);
		return newLoc;
	}

	public void add(Point gridLoc, Boid boid) {
		cells[gridLoc.x][gridLoc.y].add(boid);
	}
	public void remove(Point oldGridLoc, Boid boid) {
		cells[oldGridLoc.x][oldGridLoc.y].remove(boid);		
	}


	public class Cell {
		Vector2f topLeft = new Vector2f();
		Vector2f botRight = new Vector2f();

		public boolean isTouched = false;

		ConcurrentHashMap<Boid, Boid> boids = new ConcurrentHashMap<Boid, Boid>();

		public Cell(int col, int row) {
			topLeft.x = ((float)col) * cellWidth;
			botRight.x = topLeft.x +cellWidth;
			topLeft.y = ((float)row) * cellHeight;
			botRight.y = topLeft.y + cellHeight;
		}

	
		public void add(Boid b) {
			boids.put(b, b);
		}
		public void remove(Boid b) {
			boids.remove(b);
		}

		public Enumeration<Boid> boids() {
			return boids.elements();
		}
	}


	public void draw(GLAutoDrawable drawable, GL gl) {
		//make grey or transparent
		float top = bounds.getTop();
		float bot = bounds.getBottom();
		float left = bounds.getLeft();
		float right = bounds.getRight();
		float x = left;
		float y = top;
		
		//gl.glColor3f(.1f, 0f, .0f);		
		gl.glColor3f(.4f, 0f, .0f);	//2012	
		for(int ix = 0; ix < cells.length; ix ++) {
			Cell[] row = cells[ix];
			for(int iy = 0; iy < row.length; iy++) {
				Cell c = row[iy];
				if(c.isTouched) {
					gl.glRectf(c.topLeft.x, c.topLeft.y, c.botRight.x, c.botRight.y);					
				}
				
			}
		}

		gl.glColor3f(.1f, .1f, .1f);		
		gl.glBegin(GL.GL_LINES);

	
		for(int i = 0; i <  cells.length; i++) {
			gl.glVertex2f(x, top);
			gl.glVertex2f(x, bot);
			x+= cellWidth;
		} 


		for(int i = 0; i <  cells[0].length; i++) {
			gl.glVertex2f(left, y);
			gl.glVertex2f(right, y);			
			y+=cellHeight;
		} 
		// shade for occupancy here

		gl.glEnd();

		

	}



	public void receivedData(byte[] data) {

		
	}
}