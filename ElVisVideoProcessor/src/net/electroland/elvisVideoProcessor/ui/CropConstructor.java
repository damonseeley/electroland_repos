package net.electroland.elvisVideoProcessor.ui;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;

import javax.media.jai.RenderedOp;
import javax.media.jai.operator.CropDescriptor;


public class CropConstructor implements MouseListener, MouseMotionListener {

	int srcWidth ;
	int srcHeight;

	public static final int HANDLE_R = 4;
	public static final int HANDLE_D = HANDLE_R+HANDLE_R;
	public static final int HANDLE_RSQ = HANDLE_R * HANDLE_R;

	public Rectangle rect ;

	int mouseX;
	int mouseY;


	public CropConstructor(int srcw, int srch, int x, int y, int w, int h) {
		srcWidth =w;
		srcHeight = h;
		rect = new Rectangle(x,y,w,h);	
	}



	public CropConstructor(int srcW, int srcH, String s) {
		srcWidth =srcW;
		srcHeight = srcH;

		String[] pnts = s.split(",");
		int x = Integer.parseInt(pnts[0]);
		int y = Integer.parseInt(pnts[1]);
		int w = Integer.parseInt(pnts[2]);
		int h = Integer.parseInt(pnts[3]);
		rect = new Rectangle(x,y,w,h);	
	}

	public void renderDrawing(Graphics2D g2d) {		
			g2d.setColor(Color.BLUE);
		g2d.draw(rect);
		drawPoint(g2d, rect.x, rect.y, true);
		drawPoint(g2d, rect.x + rect.width, rect.y, true);
		drawPoint(g2d, rect.x, rect.y+ rect.height, true);
		drawPoint(g2d, rect.x+rect.width, rect.y+rect.height, true);

	}

	public int distSq(int x, int y, int x2, int y2) {
		int dx = x-x2;
		int dy = y-y2;

		return (dx*dx)+(dy*dy);
	}

	public boolean isMouseOver(int x, int y) {
		return 			(distSq(x,y,mouseX,mouseY) < HANDLE_RSQ);

	}
	public boolean drawPoint(Graphics2D g2d, int x, int y, boolean selectionFree) {
		boolean result = false;
		if(selected==null) {
			if (isMouseOver(x,y)){
				result = true;
			} 
		}
		if(result) {
			g2d.setColor(Color.RED);			
		} else {
			g2d.setColor(Color.BLUE);						
		}
		g2d.fillOval(x-HANDLE_R, y-HANDLE_R, HANDLE_D,HANDLE_D);

		return result;
	}
	public void mouseClicked(MouseEvent e) {
	}
	public void mouseEntered(MouseEvent e) {
	}
	public void mouseExited(MouseEvent e) {
	}
	public void mousePressed(MouseEvent e) {
	}
	public void mouseReleased(MouseEvent e) {
		selected = null;

	}

	Rectangle selected = null;
	public void mouseDragged(MouseEvent e) {
		int newMouseX = e.getX();
		int newMouseY = e.getY();

			if((selected == null) || (selected == rect)) {
				if(isMouseOver(rect.x, rect.y)) {
					rect.x += newMouseX-mouseX;
					rect.y += newMouseY-mouseY;
					rect.width -= newMouseX-mouseX;
					rect.height -= newMouseY-mouseY;	
					selected = rect;
				} else if (isMouseOver(rect.x + rect.width, rect.y)) {
					rect.width += newMouseX-mouseX;
					rect.y += newMouseY-mouseY;
					rect.height -= newMouseY-mouseY;								
					selected = rect;
				} else if (isMouseOver(rect.x, rect.y+ rect.height)) {
					rect.x += newMouseX-mouseX;
					rect.height += newMouseY-mouseY;				
					rect.width -= newMouseX-mouseX;
					selected = rect;
				} else if (isMouseOver(rect.x+rect.width, rect.y+rect.height)) {
					rect.width += newMouseX-mouseX;
					rect.height += newMouseY-mouseY;								
					selected = rect;
				} 



				if(rect.width < 0) {
					rect.x += rect.width;
					rect.width = -rect.width;
				}
				if(rect.height < 0) {
					rect.y += rect.height;
					rect.height = - rect.height;
				}

				if(rect.x < 0) {
					rect.x = 0;
				}
				if(rect.y < 0) {
					rect.y = 0;
				}

				if(rect.x + rect.width > srcWidth) {
					rect.width = srcWidth - rect.x;
				}
				if(rect.y + rect.height > srcHeight) {
					rect.height = srcHeight - rect.y;
				}

		}
		mouseX = newMouseX;
		mouseY = newMouseY;


	}

	public void mouseMoved(MouseEvent e) {
		mouseX = e.getX();
		mouseY = e.getY();
		selected = null;

	}


	public String toString() {
		StringBuffer sb = new StringBuffer();
			sb.append((int)rect.getX());
			sb.append(",");
			sb.append((int)rect.getY());
			sb.append(",");
			sb.append((int)rect.getWidth());
			sb.append(",");
			sb.append((int)rect.getHeight());
			sb.append("_");
		sb.deleteCharAt(sb.length()-1); // remove last _
		return sb.toString();

	}

	
	public RenderedOp getCropOp(RenderedOp op) {
		return CropDescriptor.create(op , 
				(float)rect.x,(float) rect.y,(float) rect.width,(float) rect.height, null);
	}

	public RenderedOp getCropOp() {
		return CropDescriptor.create(new BufferedImage(srcWidth,srcHeight, BufferedImage.TYPE_BYTE_GRAY) , 
				(float)rect.x,(float) rect.y,(float) rect.width,(float) rect.height, null);

	}
}
