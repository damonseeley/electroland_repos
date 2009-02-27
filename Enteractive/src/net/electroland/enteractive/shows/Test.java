package net.electroland.enteractive.shows;

import net.electroland.enteractive.core.Raster;
import net.electroland.enteractive.core.Show;

/**
 * Initial test of Show and SyncThread
 * @author asiegel
 */

public class Test implements Show{
	
	Raster raster;

	@Override
	public void cleanUp() {
		// TODO: Draw last frame
	}

	@Override
	public Raster getRasterFrame() {
		// TODO: Draw everything in here, as this gets called each frame
		return raster;
	}

	@Override
	public void initialize() {		// pass in raster or graphic object?
		// TODO: Create a canvas to draw on (PGraphics3D or Graphics) within raster
	}

}
