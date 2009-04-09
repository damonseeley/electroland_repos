package net.electroland.noho.graphics.generators.sprites;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Vector;
import java.util.Random;

import net.electroland.elvis.regions.PolyRegion;
import net.electroland.noho.core.NoHoConfig;
import net.electroland.noho.core.fontMatrix.FontMatrix;
import net.electroland.noho.util.SimpleTimer;

public class TextExplosionSprite extends Sprite {
	
	protected FontMatrix font = NoHoConfig.fontStandard;
	protected String text;
	
	// text color
	private Color c;
	private Color origColor;
	
	// private image of text
	BufferedImage textImage;
	
	// one-time X offset value
	double newXOffset = 0.0;
	
	SimpleTimer deathTimer;
	
	float thisAlpha = 1.0f;
	long fadeTime;
	float thisScale = 1.0f;
	

	/**
	 * @param x1 - initial x coordinate
	 * @param y1 - initial y coordinate
	 * @param fadeTime - how long to fade out over (ms)
	 * @param color - color of the rectangel
	 * @param text - the text to display
	 */
	public TextExplosionSprite(double x, double y, long fadeTime, Color color, String text) {
		
		super(x, y, text.length()*10, 14);
		this.c = color;
		this.origColor = color;
		this.text = text;
		this.fadeTime = fadeTime;
		// TODO Auto-generated constructor stub
		
		// make sure the text matches a physical character
		this.setX( ( ((int)(this.getX()/10)) * 10));
		
		setImage(new BufferedImage((int)getWidth(), (int)getHeight(), BufferedImage.TYPE_INT_ARGB));
		
		deathTimer = new SimpleTimer((long)fadeTime);
		
		generateTextImage();

		
	}

	@Override
	public void nextFrame(long time) {
		// TODO Auto-generated method stub
		
		//draw
		Graphics2D g2d = getImage().createGraphics();
		//fill with black
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
		g2d.setColor(Color.BLACK);
		g2d.fillRect(0, 0, getImage().getWidth(), getImage().getHeight());
		
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, thisAlpha));
		//g2d.setColor(c);
		
		//this.setImage(textImage);
		g2d.drawImage(textImage, 0, 0, null);
		

		// some hacky math here to get a hold and then fast fade
		thisAlpha = 1.5f - (float)deathTimer.percentDone()*1.5f;
		//thisAlpha = thisAlpha - .005f;
		//System.out.println(deathTimer.timeElapsed());
		
		//thisAlpha = 1.0f;
		
		
		if (thisAlpha <= 0 ){
			kill();
		} else if (thisAlpha > 1.0f){
			thisAlpha = 1.0f;
		}
		
		if (deathTimer.isDone()){
			kill();
		}
	
	}
	
	
	
	public void updateText (String newText){
		double textDiff = text.length() - newText.length();
		
		// hmm, this isnt working because of absolute X calcs based on x1 and x2
		// need to examine the model for doing this.
		newXOffset = this.getX() + textDiff*10.0;

		text = newText.toLowerCase();
		generateTextImage();
		setWidth(textImage.getWidth());
	}
	
	
	public void generateTextImage() {
		BufferedImage tempImage = new BufferedImage(text.length()*10,14,BufferedImage.TYPE_INT_ARGB);
		Graphics tig = tempImage.getGraphics();
		
		
		for(int i = 0; i < text.length(); i++) {
			BufferedImage bi = font.getLetterformImg(text.charAt(i));
			tig.drawImage(bi, i*10, 0, null);
		}
		
		// set color of text image
		int xOffset = 0;
		int colorInt = c.getRGB();
		for(int x= 0; x < tempImage.getWidth(); x++) {
			int curX = xOffset +x;
			for(int y = 0; y < tempImage.getHeight(); y++) {
				if(tempImage.getRGB(x, y)==-1) {
					tempImage.setRGB(curX, y, colorInt);
				} 
			}
		}
		
		textImage = tempImage;
	}
}
