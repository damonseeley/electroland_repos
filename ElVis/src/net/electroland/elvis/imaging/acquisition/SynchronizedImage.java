package net.electroland.elvis.imaging.acquisition;

import java.awt.image.RenderedImage;
import java.util.concurrent.atomic.AtomicReference;


public class SynchronizedImage<ImageType extends RenderedImage> {
	protected SimpleImageObserver observer;
	
	
	protected ImageType buffer;
	AtomicReference<ImageType> image;

	public SynchronizedImage(SimpleImageObserver observer) {
		this.observer = observer;
		image = new AtomicReference<ImageType> ();
	}
	
	public  void set(ImageType img) {
		image.set(img);
		observer.imageUpdated();
	}
	
	public ImageType get() {
		return image.get();
	}
	

}
