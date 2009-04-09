package net.electroland.noho.graphics.generators.sprites;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.Random;
import java.util.Vector;

import net.electroland.elvis.regions.PolyRegion;
import net.electroland.noho.core.NoHoConfig;
import net.electroland.noho.core.fontMatrix.FontMatrix;
import net.electroland.noho.util.SimpleTimer;



public class TextMotionSprite extends Sprite implements SpriteImageGeneratorListener {

	protected FontMatrix font = NoHoConfig.fontStandard;
	protected String text;
	
	// x1, y1 are the start coordinates.  x2, y2 are the end coordinates
	private double x1, y1, x2, y2;

	// totalTime is the time that this Sprite has been given to move from
	// (x1, y1) to (x2, y2)
	private long totalTime;	
	
	// rectangle color
	private Color c;
	private Color origColor;
	
	// private image of text
	BufferedImage textImage;
	
	// save the initial text
	String origText;
	
	// one-time X offset value
	double newXOffset = 0.0;
	
	// "uncollide" based on a dumb timeout
	private SimpleTimer uncollide = new SimpleTimer(1);
	
	// dummy value to start as a text counter
	double startNum = 0;
	double inc = 0.0;
	
	// need to have access to spriteWorld for explosion sprites
	SpriteImageGenerator spriteWorld;
	

	/**
	 * @param x1 - initial x coordinate
	 * @param y1 - initial y coordinate
	 * @param color - color of the rectangel
	 * @param x2 - final x coordinate
	 * @param y2 - final y coordinate
	 * @param totalTime - total time that the Sprite has been specified to move from (x1, y1) to (x2, y2).
	 */
	public TextMotionSprite(double x1, double y1,
								Color color,
								double x2, double y2,
								long totalTime,
								String text,
								SpriteImageGenerator spriteWorld) {
		
		super(x1, y1, text.length()*10, 14);
		this.x1 = x1;
		this.y1 = y1;
		this.x2 = x2;
		this.y2 = y2;
		this.totalTime = totalTime;
		this.text = text.toLowerCase();
		this.origText = this.text;
		this.c = color;
		this.origColor = color;
		this.spriteWorld = spriteWorld;
		
		setImage(new BufferedImage((int)getWidth(), (int)getHeight(), BufferedImage.TYPE_INT_ARGB));
		
		inc = new Random().nextDouble() * new Random().nextDouble() + .25;
		//inc = 1.1;
		//System.out.println("TextMotionSprite inc = " + inc);
		
		//generate the textImage
		generateTextImage();
	}

	@Override
	public void nextFrame(long time) {
		if (time - getStartTime() >= totalTime){
			// once we've hit our destination, mark us for gc.
			kill();
		}else{
			// render it
			
			// calculate it's progress
			double percentDone = (time - getStartTime()) / (double)totalTime;
			double tempX = (percentDone * (x2 - x1)) + x1 - this.getWidth()/2;
			
			//check to see if it's uncollided
			if (uncollide.isDone()) {
				c = origColor;
				
				startNum += inc;
				//startNum = tempX;
				int tmpInt = (int)startNum;
				//long timeActive = (System.currentTimeMillis() - this.getStartTime())/5;
				updateText(  ((Object)tmpInt).toString()  );
			}

			
			//Graphics2D g2d = getImage().createGraphics();

			//g2d.setColor(c);
			//g2d.draw(new Rectangle(0,0,(int)getWidth() - 1, (int)getHeight() - 1));
			
			//g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
			//g2d.drawImage(textImage, 0, 0, Color.BLACK, null);
			
			this.setImage(textImage);



			// make sure the text matches a physical character
			this.setX( ( ((int)(tempX/10)) * 10));
			this.setY((percentDone * (y2 - y1)) + y1);
			
			
			//this.setTheta(getTheta() + dtheta);
		}
	}
	
	public void generateTextImage() {
		BufferedImage tempImage = new BufferedImage(text.length()*10,14,BufferedImage.TYPE_INT_ARGB);
		Graphics tig = tempImage.getGraphics();
		
		tig.setColor(Color.BLACK);
		tig.fillRect(0, 0, text.length()*10, 14);
		
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
	
	
	public void updateText (String newText){
		double textDiff = text.length() - newText.length();
		
		// hmm, this isnt working because of absolute X calcs based on x1 and x2
		// need to examine the model for doing this.
		newXOffset = this.getX() + textDiff*10.0;

		
		text = newText.toLowerCase();
		generateTextImage();
		setWidth(textImage.getWidth());
	}
	
	
	
	public boolean isLeftToRight (){
		if (x2 > x1){
			return true;
		} else {
			return false;
		}
	}

	/* (non-Javadoc)
	 * 
	 * LinearMotionSprites are simply going to be killed off when the leave the
	 * scene.
	 * 
	 * @see net.electroland.noho.graphics.generators.sprites.SpriteAnimatorListener#exitDetected(net.electroland.noho.graphics.generators.sprites.Sprite, net.electroland.noho.graphics.generators.sprites.SpriteAnimator)
	 */
	public void exitDetected(Sprite exited, SpriteImageGenerator animator) {
		if (isDead()){
//			animator.removeListener(this);
		}else{			
			if (exited == this){
				// do nothing for now. (DO NOT CALL kill() HERE.  we may be starting
				// sprites off screen as a compensation method, so we'd be killing
				// sprites before they started
			}
		}
	}
	

	/* (non-Javadoc)
	 * 
	 * Generate a collision animation. Currently increases rotational velocity
	 * and deflects each object into different directions.
	 * 
	 * @see net.electroland.noho.graphics.generators.sprites.SpriteAnimatorListener#overlapDetected(net.electroland.noho.graphics.generators.sprites.Sprite, net.electroland.noho.graphics.generators.sprites.Sprite, net.electroland.noho.graphics.generators.sprites.SpriteAnimator)
	 */
	public void overlapDetected(Sprite one, Sprite two, SpriteImageGenerator animator) {
		if (isDead()){
//			animator.removeListener(this);
		}else{
			
			// when creating explosion sprites only update a single explosion
			// how to create a single explosion but change both text sprites to !!!?
			// if (one == this || two == this){
			if (one == this){
				
				if (one instanceof TextMotionSprite && two instanceof TextMotionSprite) {

					if (((TextMotionSprite)one).isLeftToRight() != ((TextMotionSprite)two).isLeftToRight()) {
						

						//CREATE STATIC TEXT EXPLOSION SPRITE
						
						Vector<String> punct = new Vector<String>();
						punct.add("!"); punct.add("@"); punct.add("#"); punct.add("$"); punct.add("%"); punct.add("&"); punct.add("*");
						punct.add("+"); punct.add("?");
						
						String punctString = punct.get(new Random().nextInt(punct.size())) + "" + punct.get(new Random().nextInt(punct.size())) + "" + punct.get(new Random().nextInt(punct.size()));
						
						TextExplosionSprite sprite;
						if (uncollide.isDone()){
							//if (this.isLeftToRight()){
								sprite = new TextExplosionSprite(this.getX()+this.getWidth()/2,this.getY(),4000,Color.RED,punctString);
								//sprite = new TextExplosionSprite(this.getX()+this.getWidth()/2,this.getY(), 3, 3,Color.WHITE);
							//} else {
								//sprite = new ExplosionSprite(this.getX()-this.getWidth()/2,this.getY(), 3, 3,Color.WHITE);
							//}
							
							spriteWorld.addSprite(sprite);
						}
						
						
						//c = Color.RED;
						//updateText("!!!");
						uncollide = new SimpleTimer(300);
						
												
					}
				}
				 

			}
			//deflect one up
			if (one == this){
				//y2 = height;
			}
			// deflect the other down
			if (two == this){
				//y2 = -1.0 * height;
			}			
		}
	}
}