package net.electroland.elvisVideoProcessor.ui;

import java.awt.Rectangle;
import java.awt.image.ImageConsumer;
import java.awt.image.ImageProducer;
import java.util.Vector;

public class MosaicProducer implements ImageProducer {
	Vector<ImageConsumer> consumers = new Vector<ImageConsumer>();
	Rectangle rect;
	
	public MosaicProducer(Rectangle rect) {
		this.rect = rect;
	}
	
	public void setSource()

	public void addConsumer(ImageConsumer ic) {
		consumers.add(ic);
	}

	public boolean isConsumer(ImageConsumer ic) {
		return consumers.contains(ic);
	}

	public void removeConsumer(ImageConsumer ic) {
		consumers.remove(ic);
		
	}

	public void requestTopDownLeftRightResend(ImageConsumer ic) {
		// TODO Auto-generated method stub
		
	}

	public void startProduction(ImageConsumer ic) {
		// TODO Auto-generated method stub
		
	}
	

}
