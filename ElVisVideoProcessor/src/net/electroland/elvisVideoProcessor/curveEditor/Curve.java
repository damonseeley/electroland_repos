package net.electroland.elvisVideoProcessor.curveEditor;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.CubicCurve2D;
import java.awt.geom.Point2D;
import java.util.Stack;
import java.util.Vector;


public class Curve {
	public static enum SELECTED { none, p1, p2, c1, c2 };

	public static int handelR = 5;
	public static int handelD = handelR * 2;
	public static int handelRSqr = handelR * handelR;


	CubicCurve2D cubcurve;
	SELECTED selected = SELECTED.none;
	Point2D midPoint = new Point2D.Double(-10,-10);

	public SELECTED getSelected() {
		return selected;
	}
	public Curve(CubicCurve2D cubcurve) {
		this.cubcurve = cubcurve;
		cubcurve.subdivide(left, right);
		if(left != null) {
			midPoint = left.getP2();
		}

	}

	CubicCurve2D left = new CubicCurve2D.Double(0,0,0,0,0,0,0,0);
	CubicCurve2D right = new CubicCurve2D.Double(0,0,0,0,0,0,0,0);

	public void setCurve(Point2D p1, Point2D c1, Point2D c2, Point2D p2) {
		cubcurve.setCurve(p1,c1,c2,p2);
		cubcurve.subdivide(left, right);
		if(left != null) {
			midPoint = left.getP2();
		}
	}

	public boolean render(Graphics2D g, int mouseX, int mouseY, int lastX, int lastY, boolean mouseDown) {
		g.setColor(Color.BLACK);
		g.draw(cubcurve);
		g.setColor(Color.DARK_GRAY);    	
		g.drawLine((int)cubcurve.getCtrlX1(), (int)cubcurve.getCtrlY1(), (int)cubcurve.getX1(), (int)cubcurve.getY1());
		g.drawLine((int)cubcurve.getCtrlX2(), (int)cubcurve.getCtrlY2(), (int)cubcurve.getX2(), (int)cubcurve.getY2());


		if(selected !=SELECTED.none) {
			if(mouseDown) {
				int dx = mouseX - lastX;
				int dy = mouseY - lastY;

				switch(selected) {
				
				case p1:
					setCurve(move(cubcurve.getP1(), dx,dy), cubcurve.getCtrlP1(), cubcurve.getCtrlP2(), cubcurve.getP2());
					return (renderIfOver(g, cubcurve.getP1(), mouseX, mouseY)) ;
				case p2:
					setCurve(cubcurve.getP1(), cubcurve.getCtrlP1(), cubcurve.getCtrlP2(),move(cubcurve.getP2(), dx,dy));
					return (renderIfOver(g, cubcurve.getP2(), mouseX, mouseY)) ;
				case c1:
					setCurve(cubcurve.getP1(),move(cubcurve.getCtrlP1(), dx,dy), cubcurve.getCtrlP2(), cubcurve.getP2());

					return(renderIfOver(g, cubcurve.getCtrlP1(), mouseX, mouseY)) ;
				case c2:
					setCurve(cubcurve.getP1(), cubcurve.getCtrlP1(), move(cubcurve.getCtrlP2(), dx,dy), cubcurve.getP2());
					return (renderIfOver(g, cubcurve.getCtrlP2(), mouseX, mouseY)) ;
				}

				return false; // not really reachable
			} else {
				selected = SELECTED.none;
				return false;
			}
		} else {
			if (renderIfOver(g, cubcurve.getCtrlP1(), mouseX, mouseY)) {
				if(mouseDown) {
					selected = SELECTED.c1;				
					return true; 
				}
			}
			if (renderIfOver(g, cubcurve.getCtrlP2(), mouseX, mouseY)) {
				if(mouseDown) {
					selected = SELECTED.c2;
					return true; 
				}
			}
			if (renderIfOver(g, cubcurve.getP1(), mouseX, mouseY)) {
				if(mouseDown) {
					selected = SELECTED.p1;
					return true; 
				}
			}
			if (renderIfOver(g, cubcurve.getP2(), mouseX, mouseY)) {
				if(mouseDown) {
					selected = SELECTED.p2;
					return true; 
				}
			}
			renderIfOver(g, midPoint, mouseX, mouseY, Color.DARK_GRAY,Color.GRAY);
			return false;
		}
	}
	public boolean renderIfOver(Graphics2D g, Point2D pt, int mouseX, int mouseY) {
		return renderIfOver(g,pt,mouseX,mouseY, Color.RED, Color.BLACK);
	}

	public boolean renderIfOver(Graphics2D g, Point2D pt, int mouseX, int mouseY, Color highlight, Color regularColor) {
		if (pt.distanceSq(mouseX, mouseY) < handelRSqr) {
			g.setColor(highlight);    	
			g.fillOval((int) pt.getX() - handelR,(int) pt.getY() - handelR, handelD, handelD);
			return true;
		} else {
			g.setColor(regularColor);    	
			g.fillOval((int) pt.getX() - 2,(int) pt.getY() - 2, 4, 4);
			return false;
		}
	}

	public Point2D move(Point2D pt, int dx, int dy) {
		double x = pt.getX() + dx;
		double y = pt.getY() + dy;
		pt.setLocation(x, y);
		return pt;

	}
	public Vector<Point2D> getPoints(double accuracySqr) {
		Stack<CubicCurve2D> stack = new Stack<CubicCurve2D>();
		stack.push(cubcurve);
		Vector<Point2D> points = new Vector<Point2D>();
		while(! stack.isEmpty()) {
			CubicCurve2D curve = stack.pop();
			if(curve.getP1().distanceSq(curve.getCtrlP2()) < accuracySqr) {
				Point2D pt = new Point2D.Double(
						(curve.getP1().getX() + curve.getP2().getX()) *.5,
						(curve.getP1().getY() + curve.getP2().getY()) *.5);
				points.add(pt);
						
			} else {
				CubicCurve2D left = new CubicCurve2D.Float(0,0,0,0,0,0,0,0);
				CubicCurve2D right = new CubicCurve2D.Float(0,0,0,0,0,0,0,0);
				curve.subdivide(left, right);
				if(left == null) {
					points.add(curve.getP1());
				}
				if(right == null) {
					points.add(curve.getP2());
				}
				
				if(left.equals(curve)) {
					points.add(curve.getP1());
					points.add(curve.getP2());					
				} else if(right.equals(curve)) {
					points.add(curve.getP1());
					points.add(curve.getP2());										
				} else {
					if(left != null) {
						stack.add(left);
					}
					if(right != null) {
						stack.add(right);
					}
				}

			}
		}
		return points;
		
	}
}
