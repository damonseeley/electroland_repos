package net.electroland.fish.core;

import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.FishProps;

public interface PondGenerator {
	

	public void generate(FishProps props, Pond pond) throws Exception;
	
	public void setStartPosition(Boid b, Bounds bounds) ;
}
