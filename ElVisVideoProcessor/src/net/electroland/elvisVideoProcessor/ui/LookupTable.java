package net.electroland.elvisVideoProcessor.ui;

import java.awt.image.renderable.ParameterBlock;

import javax.media.jai.JAI;
import javax.media.jai.LookupTableJAI;
import javax.media.jai.RenderedOp;

public class LookupTable {

	RenderedOp lookupOp;

	public LookupTable(RenderedOp src) {

		short[] lut = new short[65536];
		
		for(int i = 0; i < lut.length; i++) {
				lut[i] = (short) i; // identity
		}
		
		createLookup(src, lut);
	}
	
	public LookupTable(RenderedOp src, int[] ar) {

		short[] lut = new short[65536];
		
		for(int i = 0; i < lut.length; i++) {
				lut[i] = (short) ar[i];
		}
		
		createLookup(src, lut);

	}
	public void createLookup(RenderedOp src, short[] lut) {
		LookupTableJAI lookup = new LookupTableJAI(lut,0,true);
		ParameterBlock pb = new ParameterBlock();
		pb.addSource(src);
		pb.add(lookup);
		lookupOp = JAI.create("lookup", pb, null);
		
	}
 

	public RenderedOp getLookupOp() {
		return lookupOp;
	}
}