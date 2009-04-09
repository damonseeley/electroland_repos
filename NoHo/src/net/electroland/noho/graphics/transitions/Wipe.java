package net.electroland.noho.graphics.transitions;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.graphics.Transition;

public class Wipe extends Transition {

	public static enum Direction {
		TOP_TO_BOTTOM, BOTTOM_TO_TOP, LEFT_TO_RIGHT, RIGHT_TO_LEFT
	}

	public Direction direction = Direction.LEFT_TO_RIGHT;

	BufferedImage channel2preRender;

	public Wipe(int width, int height) {
		super(width, height);
		// channel2preRender = new BufferedImage(width, height,
		// BufferedImage.TYPE_INT_ARGB);

	}

	protected void drawChannel(long dt, long curTime, Graphics2D g2d,
			Channel c, int x, int y, int x2, int y2) {
		if (c != null) {
			c.ig.nextFrame(dt, curTime);
			g2d.drawImage(c.image, x, y, x2, y2, x, y, x2, y2, null);
		}
	}

	//TODO: do we need to  fade in last row of pixels for smoother wipes?   
	
	@Override
	protected void renderTransition(long dt, long curTime, double interpVal,
			Graphics2D g2d) {
		int w = image.getWidth();
		int h = image.getHeight();
		
		int wipe;
		
		switch (direction) {
			case LEFT_TO_RIGHT: {
				wipe = (int) ((1.0 - interpVal) * w);
				drawChannel(dt, curTime, g2d, channel1, wipe, 0, w, h);
				drawChannel(dt, curTime, g2d, channel2, 0, 0, wipe, h);
			} break;
			case RIGHT_TO_LEFT: {
				wipe = (int) (interpVal * w);
				drawChannel(dt, curTime, g2d, channel1, 0, 0, wipe, h);
				drawChannel(dt, curTime, g2d, channel2, wipe, 0, w, h);
			} break;
			case TOP_TO_BOTTOM: {
				wipe = (int) ((1.0 - interpVal) * h);
				drawChannel(dt, curTime, g2d, channel1, 0, wipe, w, h);
				drawChannel(dt, curTime, g2d, channel2, 0, 0, w, wipe);	
			} break;
			case BOTTOM_TO_TOP: {
				wipe = (int) (interpVal * h);
				drawChannel(dt, curTime, g2d, channel1, 0, 0, w, wipe);
				drawChannel(dt, curTime, g2d, channel2, 0, wipe, w, h);	
			} break;
		}
//		int w = image.getWidth();
//		int wipeX = (int) (interpVal * w);
//		int h = image.getHeight();


	}

	@Override
	public void reset() {
		interp.reset();
		
	}

}
