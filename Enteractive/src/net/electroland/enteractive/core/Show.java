package net.electroland.enteractive.core;

/**
 * Interface for creating show designs.
 */

public interface Show {
	// TODO: need to have a Model type to properly implement these
	//abstract public Raster getRasterFrame(Model m);
	//abstract public void initialize(Model m);
	//abstract public void cleanUp(Model m);
	
	// TODO: remove these place holders when Model is created
	abstract public Raster getRasterFrame();
	abstract public void initialize();
	abstract public void cleanUp();
}
