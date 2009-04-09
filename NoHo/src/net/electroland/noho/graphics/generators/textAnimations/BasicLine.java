package net.electroland.noho.graphics.generators.textAnimations;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.core.NoHoConfig;
import net.electroland.noho.core.fontMatrix.FontMatrix;
import net.electroland.noho.graphics.ImageGenerator;

/**
 * displays a single line of text
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class BasicLine extends ImageGenerator {
	boolean isDone = false;

	protected int charWidth = NoHoConfig.CHARWIDTH;
	protected int displayWidth = NoHoConfig.DISPLAYWIDTH;
	protected FontMatrix font = NoHoConfig.fontStandard;
	
	protected long displayTime = 5 * 1000;
	protected long changeTime = 0;

	protected BufferedImage cachedImage;	// prerender image of current line
	
	public static enum Alignment { LEFT, RIGHT, CENTER };
	
	protected Alignment alignment = Alignment.CENTER;
	
	protected int color = Color.WHITE.getRGB();
	
	
	
	public BasicLine(int width, int height) {
		super(width, height);
		cachedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	}

	@Override
	public void render(long dt, long curTime) {

		Graphics2D g2d = image.createGraphics();
		clearBackground(g2d);
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		image.createGraphics().drawImage(cachedImage,0,0, null);	
		
		if(! isDone) {
			if(changeTime <= 0) {
				changeTime = curTime + displayTime - dt; // int first frame

			} 
			if(changeTime <= curTime) {
				isDone = true;
			}
			
		}
		
	}
	
	public void prerenderLine(String s) {
		Graphics2D g2d = cachedImage.createGraphics();
		
		if(bgColor == null) {
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
			g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
		} else {
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
			g2d.setColor(bgColor);
			g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
		}
		
		int xOffset = 0; //TODO: do we need padding? (IE do charaters look ok flush left and right?)
		if(alignment != Alignment.LEFT) {
			int lineWidth = s.length() * charWidth;
			xOffset = displayWidth - lineWidth; //now flush right
			if(alignment == Alignment.CENTER) {
				xOffset *= .5;
				if (s.length()%2 == 0){
					xOffset -= 5;					
				}
//				xOffset = ((int)(xOffset / 10))*10;
			}
		}

		for(int i = 0; i < s.length(); i++) {
			BufferedImage bi = font.getLetterformImg(s.charAt(i));
			for(int x= 0; x < bi.getWidth(); x++) {
				int curX = xOffset +x;
				for(int y = 0; y < bi.getHeight(); y++) {
					if(bi.getRGB(x, y)==-1) {
						cachedImage.setRGB(curX, y, color);
					} 
				}
			}
			
			xOffset += charWidth;
		}
		
		
	}
	public void setCharWidth(int i) {
		charWidth = i;
	}
	
	public void setFontMatrix(FontMatrix font) {
		this.font = font;
	}
	
	public void setDisplayTime(long displayTime) {
		this.displayTime = displayTime;
		changeTime = 0;
	}
	
	public void setColor(Color c) {
		color = c.getRGB();
	}

	@Override
	public boolean isDone() {
		return isDone;
	}


	public void reset() {
		System.err.println("Can not reset basic line");
	}
	
	

}
