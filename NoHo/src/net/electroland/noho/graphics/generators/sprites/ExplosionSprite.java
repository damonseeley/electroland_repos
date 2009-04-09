package net.electroland.noho.graphics.generators.sprites;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.Vector;
import java.util.Random;

import net.electroland.elvis.regions.PolyRegion;
import net.electroland.noho.util.SimpleTimer;

public class ExplosionSprite extends Sprite {
	
	Color c;
	SimpleTimer deathTimer;
	float thisAlpha = 1.0f;
	float thisScale = 1.0f;
	private Vector <Rectangle>rectangles = new Vector<Rectangle>();
	

	public ExplosionSprite(double x, double y, double w, double h, Color c) {
		
		super(x, y, w, h);
		this.c = c;
		// TODO Auto-generated constructor stub
		
		setImage(new BufferedImage((int)getWidth(), (int)getHeight(), BufferedImage.TYPE_INT_ARGB));
		
		rectangles.add(new Rectangle(0,0,(int)w,(int)h));
		
		deathTimer = new SimpleTimer(4000);
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
		g2d.setColor(c);
		
		for(Rectangle rect : rectangles) {
//			/g2d.fill(rect);		
			//g2d.draw(new Rectangle(1,1,(int)getWidth() - 3, (int)getHeight() - 3));		
			//g2d.draw(new Rectangle(2,2,(int)getWidth() - 5, (int)getHeight() - 5));	
		}
		
		g2d.fill(rectangles.get(0));		
		
		//this.setWidth(50);
		
//		if (new Random().nextInt(100) > 95) {
//			xOffset += 5;
//			rectangles.add(new Rectangle(0+xOffset,0,9,9));
//		
//			//System.out.println(this.getWidth());
//			//this.setWidth(xOffset);
//		}
		
		//then scale the sprite
		//eg height*=1.5;
		
//		thisScale *= 1.01f;
//		this.setHeight(this.getHeight()*thisScale);
//		this.setWidth(this.getWidth()*thisScale);
		//this.setImage(this.getImage());
		
//		System.out.println(this + "    " + this.getX() + ":    " + thisScale);
		
		this.setTheta(this.getTheta() + 1);
		
		
		thisAlpha -= .01;
		if (thisAlpha <= 0 ){
			kill();
		}
		
		if (deathTimer.isDone()){
			kill();
		}
	
	}
}
