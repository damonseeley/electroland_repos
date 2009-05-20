package net.electroland.elvisVideoProcessor.ui;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;

import javax.media.jai.RenderedOp;

import net.electroland.elvisVideoProcessor.util.SwapBuffer;


public class MosaicConstructor implements MouseListener, MouseMotionListener {

	int srcWidth ;
	int srcHeight;

	public static final int HANDLE_R = 4;
	public static final int HANDLE_D = HANDLE_R+HANDLE_R;
	public static final int HANDLE_RSQ = HANDLE_R * HANDLE_R;

	Rectangle[] rects ;

	int mouseX;
	int mouseY;

	SwapBuffer<BufferedImage[]> swapBuffer;
	BufferedImage[]  singleLine;
	


	public MosaicConstructor(int w, int h, int outW, int outH, int rectCnt) {
		srcWidth =w;
		srcHeight = h;
		rects = new Rectangle[rectCnt];

		int boxSize = h / (3*rectCnt);
		for(int i = 0; i < rectCnt; i++) {
			rects[i] = new Rectangle(10, (3*i*boxSize) + boxSize, w-20,boxSize);
		}

		rebuildResultPool(outW, outH);
	}



	public MosaicConstructor(int srcW, int srcH, int outW, int outH, String s) {
		srcWidth =srcW;
		srcHeight = srcH;

		String[] rectStrings = s.split("_");
		rects = new Rectangle[rectStrings.length];
		int i = 0;
		for(String rectString : rectStrings) {
			String[] pnts = rectString.split(",");
			int x = Integer.parseInt(pnts[0]);
			int y = Integer.parseInt(pnts[1]);
			int w = Integer.parseInt(pnts[2]);
			int h = Integer.parseInt(pnts[3]);
			rects[i++] = new Rectangle(x,y,w,h);			
		}
		rebuildResultPool(outW, outH);
	}

	public void rebuildResultPool(int outWidth, int outHeight) {

		BufferedImage[] resultCache = new BufferedImage[rects.length];
		BufferedImage[] resultCache2 = new BufferedImage[rects.length];
		BufferedImage[] singleLine = new BufferedImage[rects.length];
		for(int i = 0; i < rects.length; i++) {
			resultCache[i] = new BufferedImage(outWidth, outHeight, BufferedImage.TYPE_USHORT_GRAY);
			resultCache2[i] = new BufferedImage(outWidth, outHeight, BufferedImage.TYPE_USHORT_GRAY);
			singleLine[i] = new BufferedImage(outWidth, 1, BufferedImage.TYPE_USHORT_GRAY);
		}
		this.singleLine = singleLine;
		swapBuffer = new SwapBuffer<BufferedImage[]>(resultCache,resultCache2);

	}
	public void renderDrawing(Graphics2D g2d) {		
		boolean rectFree = true;
		for(Rectangle rect : rects) {
			if(selected == rect || (selected == null && rect.contains(mouseX,mouseY) ) && rectFree) {
				g2d.setColor(Color.RED);
				rectFree = false;
			} else {
				g2d.setColor(Color.BLUE);
			}
			g2d.draw(rect);
			drawPoint(g2d, rect.x, rect.y, true);
			drawPoint(g2d, rect.x + rect.width, rect.y, true);
			drawPoint(g2d, rect.x, rect.y+ rect.height, true);
			drawPoint(g2d, rect.x+rect.width, rect.y+rect.height, true);

		}
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

		for(Rectangle rect : rects) {
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
				} else if(rect.contains(mouseX, mouseY)) {
					rect.x += newMouseX-mouseX;
					rect.y += newMouseY-mouseY;
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
		for(Rectangle rect : rects) {
			sb.append((int)rect.getX());
			sb.append(",");
			sb.append((int)rect.getY());
			sb.append(",");
			sb.append((int)rect.getWidth());
			sb.append(",");
			sb.append((int)rect.getHeight());
			sb.append("_");
		}
		sb.deleteCharAt(sb.length()-1); // remove last _
		return sb.toString();

	}

	public void processImage(RenderedOp input) {
		BufferedImage[] resultCache;
		try {
			resultCache = swapBuffer.takeToProcess();
			BufferedImage img = input.getAsBufferedImage();
			BufferedImage[] ar = new BufferedImage[rects.length];
			for(int i = 0; i < ar.length; i++) {
				BufferedImage subImage = img.getSubimage(rects[i].x, rects[i].y, rects[i].width, rects[i].height);
				float lineHight  = img.getHeight() / (float) resultCache[i].getHeight();
				for(int line = 0; line < resultCache[i].getHeight(); line++) {
					singleLine[i].createGraphics().drawImage(subImage, 0, line, singleLine[i].getWidth(), singleLine[i].getHeight(), 0,(int)(line * lineHight), subImage.getWidth(), (int) lineHight, null);
					resultCache[i].createGraphics().drawImage(singleLine[i], 0, (int)(line * lineHight), resultCache[i].getWidth(), (int)lineHight, 0,0, singleLine[i].getWidth(), singleLine[i].getHeight(), null);					
				}
//				resultCache[i] = singleLine[i];
//				resultCache[i].createGraphics().drawImage(singleLine[i], 0, 0, singleLine[i].getWidth()-1, 0, 0,0, resultCa//che[i].getWidth()-1, 6, null);
			}
			swapBuffer.putProcessed(resultCache);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}

	public BufferedImage[] getImage() throws InterruptedException {
		return swapBuffer.takeProcessed();
	}


}
