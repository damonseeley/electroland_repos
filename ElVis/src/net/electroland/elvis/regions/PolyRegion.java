package net.electroland.elvis.regions;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Stroke;
import java.io.Serializable;
import java.util.StringTokenizer;
import java.util.Vector;

import net.electroland.elvis.imaging.RoiAve;
import net.electroland.elvis.regionManager.ImagePanel;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class PolyRegion extends BasePolyRegion implements Serializable {


	transient Vector<TriggerListener> listeners = new Vector<TriggerListener>();

	public static final double CLAMP_SCALE_VALUE = 1.0 / (double) 255;


	public static int CUR_ID = 0;



	transient RoiAve roiAve = null;


	public boolean isSelected = false;
	public static float COLOR_SCALE = 1.0f / 255.0f;

	int depth = 0;

	public boolean isFilled = false;
	Color baseColor = Color.RED;
	Color lineColor = Color.RED;
	Color fillColor = new Color(1,0,0,.25f);
	Color selectedColor = new Color(1,0,0,.25f);
	Color triggerColor = new Color(1,0,0,.75f);
	Color triggerStrokeColor = new Color(1,0,0,.75f);
	transient BasicStroke triggerStroke = new BasicStroke(10);

	protected Polygon poly = new Polygon();


	public PolyRegion() {
		this(CUR_ID++, "regaion_" + CUR_ID, .5f);
	}

	public PolyRegion(int id, String name, Polygon p, float percentage) {
		this(id,  name, percentage);
		poly = p;
		this.updateROI();
	}

	public PolyRegion(int id, String name, float percentage) {
		super(id, false, -1, name, percentage);
	}



	public boolean isTriggered(IplImage ri) {
		mean = roiAve.getAverage(ri) * CLAMP_SCALE_VALUE;

		isTriggered = mean >= percentage;
		if(isTriggered) {
			for(TriggerListener tr : listeners) {
				tr.trigger(this);
			}
		}
		return isTriggered;

	}
//	public PolyRegion(int id, String name, Polygon p) {

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(name);
		sb.append(",");
		sb.append(poly.npoints);
		sb.append(",");
		for(int i = 0; i < poly.npoints; i++) {
			sb.append(poly.xpoints[i]);
			sb.append(",");
		}
		for(int i = 0; i < poly.npoints; i++) {
			sb.append(poly.ypoints[i]);
			sb.append(",");
		}
		sb.append(percentage);
		sb.append(",");		
		sb.append(id);
		return sb.toString();
	}
	public static PolyRegion fromString(String s) {
		StringTokenizer t = new StringTokenizer(s, ",");
		String name = t.nextToken();
		int nPoints = Integer.parseInt(t.nextToken());
		int[] xPoints = new int[nPoints];
		int[] yPoints = new int[nPoints];
		for(int i = 0; i < nPoints; i++) {
			xPoints[i] =  Integer.parseInt(t.nextToken());
		}
		for(int i = 0; i < nPoints; i++) {
			yPoints[i] =  Integer.parseInt(t.nextToken());
		}
		float perc = Float.parseFloat(t.nextToken());

		int id = Integer.parseInt(t.nextToken());
		
		Polygon p = new Polygon(xPoints, yPoints, nPoints);
		
		PolyRegion poly =  new PolyRegion( id, name, p, perc); 
		poly.isFilled = true;
		return poly;
	}

	public int getDepth(){
		return depth;
	}
	public void setDepth(int d) {
		depth = d;
	}

	public Color getColor() {
		return baseColor;
	}
	public void addPoint(int x, int y) {
		poly.addPoint(x, y);
		updateROI();

	}

	public void translate(int deltaX, int deltaY) {
		poly.translate(deltaX, deltaY);
		updateROI();

	}
	public boolean contains(int x, int y) {
		return poly.contains(x, y);
	}

	public void setColor(Color c) {
		setColor(c.getRed()*COLOR_SCALE, c.getGreen()*COLOR_SCALE, c.getBlue()*COLOR_SCALE);
	}
	public void setColor(float r, float g, float b) {
		baseColor  = new Color(r,g,b);		
		fillColor = new Color(r,g,b,.15f);
		lineColor = new Color(r,g,b, .3f);
		selectedColor = new Color(r,g,b,.5f);
		triggerColor = new Color(r,g,b,.8f);
		triggerStrokeColor = new Color(1-r, 1-g,1-b, 1);

	}
	public void removePoint(int x, int y) {
		Polygon newPoly = new Polygon();
		for(int i = 0; i < poly.npoints; i++) {
			if((x != poly.xpoints[i]) || (y != poly.ypoints[i])) {
				newPoly.addPoint(poly.xpoints[i], poly.ypoints[i]);
			}
		}
		poly = newPoly;
		updateROI();
	}

	public void insertPoint(int index, int x, int y) {
		Polygon newPoly = new Polygon();
		for(int i = 0; i < poly.npoints; i++) {
			if(i == index) {
				newPoly.addPoint(x, y);
			}
			newPoly.addPoint(poly.xpoints[i], poly.ypoints[i]);
		}
		poly = newPoly;
		updateROI();

	}


	public void moveLastPoint(int x, int y) {
		movePoint(poly.npoints -1, x, y);

	}

	public void movePoint(int index, int x, int y) {
		poly.xpoints[index] = x;
		poly.ypoints[index] = y;		
		poly.invalidate();
		updateROI();

	}

	public void removeLastPoint() {
		poly.npoints -= 1;
		poly.invalidate();

	}
	public void removePoint(int index) {
		Polygon newPoly = new Polygon();
		for(int i = 0; i < poly.npoints; i++) {
			if(i != index) {
				newPoly.addPoint(poly.xpoints[i],poly.ypoints[i]);
			}
		}
		poly = newPoly;
		updateROI();
	}
	public int size() {
		return poly.npoints;
	}
	public boolean startPointInRange(int x, int y, int radiusSquare) 
	{
		if (poly.npoints < 3) { // need more than 2 points
			return false;
		}
		int dx = x - poly.xpoints[0];
		dx *= dx;
		if(dx < radiusSquare) {
			int d = y - poly.ypoints[0];
			d *= d;
			d += dx;
			if(d < radiusSquare) {
				return true;
			}
		}
		return false;

	}

	public static class DistResult {
		public PolyRegion region;
		public int index;
		public float distSquared;
	}

	public DistResult getNearestEdge(int x, int y, int radius,  int radiusSqr) {
		// return null if not in bounding box
		if(x < poly.getBounds().getMinX() - radius) return null;
		if(x > poly.getBounds().getMaxX() + radius) return null;
		if(y < poly.getBounds().getMinY() - radius) return null;
		if(y > poly.getBounds().getMaxY() + radius) return null;

		float minDistSqr = radiusSqr;
		int minDistIndex = -1;

		int x1 = poly.xpoints[0];
		int y1 = poly.ypoints[0];

		for(int i = 1; i < poly.npoints; i++) {
			int x2 = poly.xpoints[i];
			int y2 = poly.ypoints[i];

			float distSqr = ptLineDistSq(x1,y1,x2,y2,x,y);
			if(distSqr<minDistSqr) {
				distSqr = minDistSqr;
				minDistIndex = i;
			}

			x1 = x2;
			y1 = y2;

		}

		int x2 = poly.xpoints[0];
		int y2 = poly.ypoints[0];
		float distSqr = ptLineDistSq(x1,y1,x2,y2,x,y);
		if(distSqr<minDistSqr) {
			distSqr = minDistSqr;
			minDistIndex = 0;
		}

		if(minDistIndex != -1) {
			DistResult result = new DistResult();
			result.region = this;
			result.distSquared = minDistSqr;
			result.index = minDistIndex;
			return result;
		}
		return null;
	}

	// taken from java imp
	public static float ptLineDistSq(int X1, int Y1,
			int X2, int Y2,
			int PX, int PY) {
		//		Adjust vectors relative to X1,Y1
		//		X2,Y2 becomes relative vector from X1,Y1 to end of segment
		X2 -= X1;
		Y2 -= Y1;
		//		PX,PY becomes relative vector from X1,Y1 to test point
		PX -= X1;
		PY -= Y1;
		float dotprod = PX * X2 + PY * Y2;
		//		dotprod is the length of the PX,PY vector
		//		projected on the X1,Y1=>X2,Y2 vector times the
		//		length of the X1,Y1=>X2,Y2 vector
		float projlenSq = (float)(dotprod * dotprod) / (float)(X2 * X2 + Y2 * Y2);
		//		Distance to line is now the length of the relative point
		//		vector minus the length of its projection onto the line
		float lenSq = PX * PX + PY * PY - projlenSq;
		if (lenSq < 0) {
			lenSq = 0;
		}
		return lenSq;
	}
	public DistResult getNearestPoint(int x, int y, int radius,  int radiusSqr) {

		// return null if not in bounding box
		if(x < poly.getBounds().getMinX() - radius) return null;
		if(x > poly.getBounds().getMaxX() + radius) return null;
		if(y < poly.getBounds().getMinY() - radius) return null;
		if(y > poly.getBounds().getMaxY() + radius) return null;


		int nearIndex = -1;
		int nearDis = radiusSqr;

		for(int i = 0; i < poly.npoints; i++) {
			int dx = x - poly.xpoints[i];
			dx *= dx;
			if(dx < nearDis) {
				int d = y - poly.ypoints[i];
				d *= d;
				d += dx;
				if(d < nearDis) {
					nearIndex = i;
					nearDis = d;
				}


			}
		}
		if(nearDis < radiusSqr) {
			DistResult dr = new DistResult();
			dr.region = this;
			dr.index = nearIndex;
			dr.distSquared = nearDis;
			return dr;
		} else {
			return null;
		}
	}

	public int getX(int i) {
		return poly.xpoints[i];
	}
	public int getY(int i) {
		return poly.ypoints[i];
	}

	public void render(Graphics2D g) {
		if(isFilled) {
			if(isTriggered) {
				g.setColor(triggerStrokeColor);
				Stroke oldStroke = g.getStroke();
				g.setStroke(triggerStroke);		
				g.drawPolygon(poly);				
				g.setStroke(oldStroke);
				g.setColor(triggerColor);
			} else if(isSelected) {
				g.setColor(selectedColor);
				isSelected = false;
			} else {
				g.setColor(fillColor);
			}
			g.fillPolygon(poly);
		} else {
			g.setColor(lineColor);
			g.drawPolyline(poly.xpoints, poly.ypoints, poly.npoints);
		}
		for(int i =0; i < poly.npoints; i++) {
			drawHighlightedPoint(g, i, 1,2); 
		}

	}

	public void drawHighlightedPoint(Graphics2D g, int i, int radius_half, int radius) {
		drawHighlightedPoint(g, poly.xpoints[i], poly.ypoints[i],radius_half,radius, lineColor);
	}

	public static void drawHighlightedPoint(Graphics2D g, int x, int y,int radius_half, int radius, Color c) {
		g.setColor(c);
		g.fillOval(x-radius_half, y-radius_half, radius, radius);
	}

	public void restoreTransients() {
		updateROI();
		listeners = new Vector<TriggerListener>();
		triggerStroke = new BasicStroke(10);
	}
	public void updateROI() {
		Polygon scaledPoly = new Polygon();
		for(int i = 0; i < poly.npoints; i++) {
			scaledPoly.addPoint((int) (getX(i)* ImagePanel.INV_SCALER), (int)(getY(i)* ImagePanel.INV_SCALER));			
		}
		if(roiAve == null) {
			roiAve = new RoiAve();
		} 
		roiAve.setRoi(scaledPoly);
	}

	public void addTriggerListener(TriggerListener tl) {
		listeners.add(tl);
	}


}
