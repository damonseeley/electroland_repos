package net.electroland.elvisVideoProcessor.ui;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.geom.Point2D;
import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.PerspectiveTransform;
import javax.media.jai.RenderedOp;
import javax.media.jai.WarpGrid;

public class WarpGridConstructor implements MouseListener, MouseMotionListener {

	public static final String LINE_BREAK = "-";
	public static final String PNT_BREAK = "_";
	int srcWidth;
	int srcHeight;

	int mouseX = -1;
	int mouseY= -1;

	int selectedCol = -1;
	int selectedRow = -1;



	public static final int handleRadius = 10;
	public static final int handleDiameter = handleRadius+handleRadius;
	public static final int handleRadiusSqr = handleRadius*handleRadius;

	Point[][] grid;

	WarpGrid warpGrid;
	RenderedOp warpOp;

	public void setSrcDims(int w, int h) {
		srcWidth = w;
		srcHeight = h;
	}


	public WarpGridConstructor(int warpGridWidth, int warpGridHeight, int imgWidth, int imgHeight) {
		srcWidth = imgWidth;
		srcHeight =imgHeight;

		grid = new Point[warpGridWidth][warpGridHeight];
		float xScale = (float) imgWidth /  ((float)warpGridWidth - 1);
		float yScale = (float) imgHeight /  ((float)warpGridHeight - 1);

		for(int x = 0; x < warpGridWidth; x++) {
			for(int y = 0; y < warpGridHeight; y++) {
				grid[x][y] = new Point(
						(int) (x * xScale),
						(int) (y * yScale));

			}
		}
//		resetEdges();

	}



	public WarpGridConstructor(String s,int imgWidth, int imgHeight) {
//		System.out.println("constructing warp grid for " + imgWidth +"x"+ imgHeight);
		srcWidth = imgWidth;
		srcHeight =imgHeight;
		String[] rows = s.split(LINE_BREAK);
		String[] fistCol = rows[0].split(PNT_BREAK);
		grid = new Point[fistCol.length][rows.length];
		for(int i = 0; i < rows.length; i++) {
//			System.out.println("row:"  + rows[0]);
			String[] cols = rows[i].split(PNT_BREAK);
			for(int j = 0; j < cols.length; j++) {
//				System.out.println("col:"  + cols[j]);
				String[] pt = cols[j].split(",");
				int x = Integer.parseInt(pt[0]);
				int y = Integer.parseInt(pt[1]);
				grid[j][i] = new Point(x,y);
			}

		}	

	}


	public void addRow(StringBuffer sb, int y) {
		for(int x = 0; x < grid.length - 1; x++) {
			sb.append(grid[x][y].x);
			sb.append(",");
			sb.append(grid[x][y].y);
			sb.append(PNT_BREAK);
		}
		sb.append(grid[grid.length - 1][y].x);
		sb.append(",");
		sb.append(grid[grid.length - 1][y].y);
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		for(int y = 0; y < grid[0].length -1; y++) {
			addRow(sb, y);
			sb.append(LINE_BREAK);
		}
		addRow(sb, grid[0].length -1);
		return sb.toString();
	}


	public float mouseDistSqr(float x, float y) {
		float dx = x - mouseX;
		dx *= dx;
		float dy = y - mouseY;
		dy*=dy;
		return dx +dy;
	}

	/*
	public void rescale(Rectangle rect) {
		float xScale = (float) rect.width /  ((float)grid.length - 1);
		float yScale = (float) rect.height /  ((float)grid[0].length - 1);

		for(int x = 0; x < grid.length ; x++) {
			for(int y = 0; y < grid[0].length ; y++) {
				grid[x][y] = new Point(
						(int) (x * xScale),
						(int) (y * yScale));

			}
		}
	}
	 */
	public void renderDrawing(Graphics2D g2d) {		
		int curColI = 0;
		Point[] curCol = grid[curColI++];
		Point[] nextCol = grid[curColI++];
		while(nextCol != null) {
			for(int i = 0; i < curCol.length -1; i++) {
				int x = curCol[i].x;
				int y = curCol[i].y;
//				if((curColI != 2) && (i != 0)) {
				float d = mouseDistSqr(x,y);
//				System.out.println(x+"," + y + "-" + mouseX + ","+ mouseY + "--->"+ d);
				if(d <=handleRadiusSqr) {
					g2d.setColor(Color.RED);
				} else {
					g2d.setColor(Color.BLUE);
				}

				g2d.drawOval(x-handleRadius, y-handleRadius, handleDiameter,handleDiameter);
				//	}
				g2d.setColor(Color.BLUE);
				g2d.drawLine(x,y, curCol[i+1].x, curCol[i+1].y);
				g2d.drawLine(x, y, nextCol[i].x, nextCol[i].y);

			}
			int x = curCol[curCol.length -1].x;
			int y = curCol[curCol.length -1].y;
			float d = mouseDistSqr(x,y);
			g2d.drawLine(x,y, nextCol[curCol.length -1].x, nextCol[curCol.length -1].y);

			if(d <=handleRadiusSqr) {
				g2d.setColor(Color.RED);
			} else {
				g2d.setColor(Color.BLUE);
			}
			g2d.drawOval(x-handleRadius, y-handleRadius, handleDiameter,handleDiameter);

			g2d.setColor(Color.BLUE);

			curCol = nextCol;
			if(curColI < grid.length) {
				nextCol = grid[curColI++];				
			} else {
				nextCol = null;	
			}

		}
		for(int i = 0; i < curCol.length -1; i++) {
			int x = curCol[i].x;
			int y = curCol[i].y;

			float d = mouseDistSqr(x,y);
			g2d.drawLine(x,y, curCol[i+1].x, curCol[i+1].y);

			if(d <=handleRadiusSqr) {
				g2d.setColor(Color.RED);
			} else {
				g2d.setColor(Color.BLUE);
			}
			g2d.drawOval(x-handleRadius, y-handleRadius, handleDiameter,handleDiameter);
			g2d.setColor(Color.BLUE);

		}		
		int x = curCol[curCol.length -1].x;
		int y = curCol[curCol.length -1].y;

		float d = mouseDistSqr(x,y);
		if(d <=handleRadiusSqr) {
			g2d.setColor(Color.RED);
		} else {
			g2d.setColor(Color.BLUE);
		}
		g2d.drawOval(x-handleRadius, y-handleRadius, handleDiameter,handleDiameter);
		g2d.setColor(Color.BLUE);


	}

	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public void mousePressed(MouseEvent e) {
		selectedCol = -1;
		selectedRow = -1;
		for(int i = 0; i < grid.length; i++) {
			for(int j = 0; j < grid[0].length; j++) {
				int x = grid[i][j].x;
				int y = grid[i][j].y;
				mouseX = e.getX();
				mouseY = e.getY();
				float d = mouseDistSqr(x,y);
				if(d <=handleRadiusSqr) {
					selectedCol =i;
					selectedRow = j;
				}


			}
		}

	}

	public void mouseReleased(MouseEvent e) {
		selectedCol = -1;
		selectedRow = -1;

	}


	public void mouseDragged(MouseEvent e) {

		if(selectedCol >= 0) {
			if(! e.isShiftDown()) {
				grid[selectedCol][selectedRow].x += e.getX() -  mouseX;
				mouseX = e.getX();
			}
			if(! e.isAltDown()) {
				grid[selectedCol][selectedRow].y += e.getY() -  mouseY;
				mouseY = e.getY();			
			}


			grid[selectedCol][selectedRow].x = (grid[selectedCol][selectedRow].x <0)? 0:grid[selectedCol][selectedRow].x;
			grid[selectedCol][selectedRow].y = (grid[selectedCol][selectedRow].y <0)? 0:grid[selectedCol][selectedRow].y;
			grid[selectedCol][selectedRow].x = (grid[selectedCol][selectedRow].x > srcWidth) ? srcWidth  : grid[selectedCol][selectedRow].x ;
			grid[selectedCol][selectedRow].y = (grid[selectedCol][selectedRow].y > srcHeight)? srcHeight : grid[selectedCol][selectedRow].y ;


//			resetEdges();
		}

	}

	public void resetEdges() {
		int minX = Integer.MAX_VALUE;
		int minY = Integer.MAX_VALUE;
		int maxX = Integer.MIN_VALUE;
		int maxY= Integer.MIN_VALUE;

		for(int i = 1; i < grid.length-1;i++) {
			for(int j = 1; j < grid[0].length-1; j++) {
				minX = grid[i][j].x < minX ? grid[i][j].x : minX;			
				minY = grid[i][j].y < minY ? grid[i][j].y : minY;			
				maxX = grid[i][j].x > maxX ? grid[i][j].x : maxX;			
				maxY = grid[i][j].y > maxY ? grid[i][j].y : maxY;			

			}
		}

		//System.out.println(minX);

		for(int i = 0; i < grid.length;i++) {
			for(int j = 0; j < grid[0].length; j++) {
				grid[i][j].x = (grid[i][j].x < minX) ? minX :  grid[i][j].x ;
				grid[i][j].y = (grid[i][j].y < minY) ? minY :  grid[i][j].y ;
				grid[i][j].x = (grid[i][j].x > maxX) ? maxX :  grid[i][j].x ;
				grid[i][j].y = (grid[i][j].y > maxY) ? maxY :  grid[i][j].y ;
			}
		}


		for(int i = 0; i < grid.length; i++) {
			grid[i][0].y = minY;
			grid[i][grid[0].length-1].y = maxY;
		}

		for(int i = 0; i < grid[0].length; i++) {
			grid[0][i].x = minX;
			grid[grid.length-1][i].x = maxX;
		}

	}


	/*
		int min = Integer.MAX_VALUE;

		for(int i = 1; i < grid.length-1; i++) {
			min = grid[i][1].y < min ? grid[i][1].y : min;			
		}

		int max = -1;
		for(int i = 1; i < grid.length-1; i++) {
			max = grid[i][grid.length-2].y > max ? grid[i][grid.length-2].y : max;			
		}

		for(int i = 0; i < grid.length; i++) {
			grid[i][grid.length-1].y = max;		
		}


		min = Integer.MAX_VALUE;
		for(int i = 1; i < grid[0].length-1; i++) {
			min = grid[1][i].x < min ? grid[1][i].x : min;			
		}

		for(int i = 0; i < grid[0].length; i++) {
			grid[0][i].x = min;		
		}

		for(int i = 0; i < grid[0].length; i++) {
			grid[1][i].x = (grid[1][i].x< min) ? min : grid[1][i].x;		
		}

		max = -1;
		for(int i = 1; i < grid[0].length-1; i++) {
			max = grid[grid[0].length-2][i].x > max ? grid[grid[0].length-2][i].x : max;			
		}
		for(int i = 0; i < grid.length; i++) {
			grid[grid.length-1][i].x = max;		
		}
	 */

//	}


	public void mouseMoved(MouseEvent e) {
		if(selectedCol <0) {
			mouseX = e.getX();
			mouseY = e.getY();

		}

	}

	public void reset() {



		int maxX = 0;
		int maxY =0;
		int minX = Integer.MAX_VALUE;
		int minY = Integer.MAX_VALUE;

		for(int i = 0; i < grid.length; i++) {
			for(int j =0; j < grid[0].length; j++) {
				maxX = (maxX > grid[i][j].x ? maxX :  grid[i][j].x );
				maxY = (maxY > grid[i][j].y ? maxY :  grid[i][j].y );
				minX = (minX < grid[i][j].x ? minX :  grid[i][j].x );
				minY = (minY < grid[i][j].y ? minY :  grid[i][j].y );
			}
		}

		float xScale = (float)srcWidth   /  ((float)grid.length - 1);
		float yScale = (float)srcHeight /  ((float)grid[0].length - 1);




//		System.out.println(cropY + ", " + cropH + ", " + yScale);

		// note to self orig is grid
		// mapped you need to construct

		PerspectiveTransform[][] transforms = new PerspectiveTransform[(grid.length -1)][(grid[0].length -1)];
		for(int i = 0; i < grid.length-1; i++) {
			for(int j = 0; j < grid[0].length-1 ; j++) {
				/*				transforms[i][j] = PerspectiveTransform.getQuadToSquare(
						grid[i][j].x, grid[i][j].y,
						grid[i][j+1].x, grid[i][j+1].y,
						grid[i+1][j].x, grid[i+1][j].y,
						grid[i+1][j+1].x, grid[i+1][j+1].y);*/
				transforms[i][j] = makeTransform(
						(i*xScale),  	 (j*yScale), 
						(i*xScale),  	 ((j+1)*yScale), 
						((i+1)*xScale), 	 (j*yScale), 
						((i+1)*xScale), 	 ((j+1)*yScale)
						,
						grid[i][j].x , grid[i][j].y,
						grid[i][j+1].x, grid[i][j+1].y ,
						grid[i+1][j].x, grid[i+1][j].y ,
						grid[i+1][j+1].x , grid[i+1][j+1].y 



				);

				/*

				System.out.println(
						grid[i][j].x+","+ grid[i][j].y+" "+
						grid[i][j+1].x+","+ grid[i][j+1].y+" "+
						grid[i+1][j].x+","+ grid[i+1][j].y+" "+
						grid[i+1][j+1].x+","+ grid[i+1][j+1].y
						+"-->"+
						(minX + (i*xScale))+","+  	(minY + (j*yScale))+" "+ 
						(minX + (i*xScale))+","+  	(minY + ((j+1)*yScale))+" "+
						(minX + ((i+1)*xScale))+","+ 	(minY + (j*yScale))+" "+
						(minX + ((i+1)*xScale))+","+ 	(minY + ((j+1)*yScale))
				);
				 */

			}
		}


		float[] transformPositions = new float[(srcWidth) * (srcHeight) * 2];

		Point2D.Float origPt = new Point2D.Float() ;
		Point2D.Float tranformedPt = new Point2D.Float() ;



		float transformWidth =  (float) (srcWidth) /  ((float)grid.length - 1);
		float transformHeight = (float) (srcHeight) /  ((float)grid[0].length - 1);

		int i = 0;



		// great a mapping for very pixel orig->new
		for(float y = 0f; y< srcHeight;y ++) {
			for(float x = 0f; x< srcWidth;x++) {
				origPt.setLocation(x,y);
				int transformJ = (int) Math.floor(((float) y) / ((float) transformHeight));
				int transformI = (int) Math.floor(((float) x) / ((float) transformWidth));
				transforms[transformI][transformJ].transform(origPt, tranformedPt);
//				tranformedPt.y= ( tranformedPt.y > cropH) ? cropH : tranformedPt.y;
				//			tranformedPt.x= ( tranformedPt.x > cropW) ? cropW : tranformedPt.x;
				transformPositions[i++] = (float) tranformedPt.getX();
				transformPositions[i++] = (float) tranformedPt.getY();


			}

		}



		warpGrid=  new WarpGrid(0, 1, srcWidth-1, 0,1, srcHeight-1, transformPositions); // use warp grid becuse this uses hardware acceleration


	}

	public PerspectiveTransform makeTransform(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4,
			double dx1, double dy1, double dx2, double dy2, double dx3, double dy3, double dx4,double dy4 ) {
	/*	System.out.println(
				x1+","+y1 +" " +
				x2+","+y2 +" " +
				x3+","+y3 +" " +
				x4+","+y4 + "--->" +
				dx1+","+dy1 +" " +
				dx2+","+dy2 +" " +
				dx3+","+dy3 +" " +
				dx4+","+dy4
		);
		System.out.println("   " + x1+","+y1 +"->" +dx1+","+dy1);
		System.out.println("   " + x2+","+y2 +"->" +dx2+","+dy2);
		System.out.println("   " + x3+","+y3 +"->" +dx3+","+dy3);
		System.out.println("   " + x4+","+y4 +"->" +dx4+","+dy4);
		*/

		return PerspectiveTransform.getQuadToQuad(x1,y1,x2,y2,x3,y3,x4,y4,dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4);

	}

	public  RenderedOp getWarpOp(Object source) {
		ParameterBlock pb = new ParameterBlock();
		pb.addSource(source);
		pb.add(warpGrid);
		warpOp =  JAI.create("warp", pb);
		return warpOp;
	}


}
