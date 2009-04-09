package net.electroland.noho.graphics.generators;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.graphics.ImageGenerator;

public class StaticImage extends ImageGenerator {

	int timeLeft;
	int holdTime;
	BufferedImage referenceImage;

	
	public StaticImage(int width, int height, BufferedImage srcImage, Color c, int holdTime) {
		super(width, height);
		this.holdTime = holdTime;
		timeLeft = holdTime;
		setReferenceImage(srcImage);
	}

	public void setReferenceImage(BufferedImage img) {
		referenceImage = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_ARGB);
		referenceImage.createGraphics().drawImage(img, 0,0, null);	
	}
	
	
	@Override
	public void render(long dt, long curTime) {
		timeLeft -= dt;
		Graphics2D g2d = image.createGraphics();
		clearBackground(g2d);
		drawBackground(g2d);
	}

	@Override
	public boolean isDone() {
		return timeLeft <= 0;
	}

	public void reset() {
		timeLeft = holdTime;
	}

}
	