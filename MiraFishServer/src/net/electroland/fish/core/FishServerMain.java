package net.electroland.fish.core;

import net.electroland.fish.ui.MainFrame;
import net.electroland.fish.util.Bounds;
import net.electroland.fish.util.FishProps;

public class FishServerMain {

	/**
	 * props file name
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		FishProps props;
		if(args.length >= 1) {
			props = FishProps.init(args[0]);
		} else {
			props = FishProps.init("FishProps.props");
		}
		
		
		Maps.loadMaps(props.getProperty("noEntryMap", "noEntry.bmp"), 
				props.getProperty("rotateMap", "rotate.bmp"), props.getProperty("rotateWeight", .5f),
				props.getProperty("edgeMap", "edges.bmp"), props.getProperty("edgeWeight", 1f));

		String generatorClassName = props.getProperty("pondGeneratorClass", "MainTable");

		PondGenerator generator = (PondGenerator) Class.forName("net.electroland.fish.generators." + generatorClassName).newInstance();

		

		int gridWidth = props.getProperty("pondGridWidth", 65);
		int gridHeight = props.getProperty("pondGridHeight", 45);


		Bounds worldBounds = FishProps.THE_FISH_PROPS.getProperty("pondBounds", new Bounds(0f, 0f, 1536f, 3072f, 2f, 0f));


		Pond pond = new Pond(worldBounds, gridWidth, gridHeight);


		generator.generate(props, pond);

		pond.startRendering();

		if(props.getProperty("showGraphics", true)) {
			new MainFrame(pond).startRendering();
		} else {
			System.out.println("Running pond headless");
		}
	}

}
