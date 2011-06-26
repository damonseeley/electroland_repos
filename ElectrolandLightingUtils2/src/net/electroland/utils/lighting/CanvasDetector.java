package net.electroland.utils.lighting;

import java.awt.Shape;
import java.util.ArrayList;
import java.util.List;

public class CanvasDetector {

	protected Shape boundary;
	protected DetectionModel model;
	protected byte latestState;
	protected List<String> tags;
	// TODO: this probably needs to be synchronized (in case multiple threads
	//       attempt to sync simultaneously)
	protected ArrayList<Integer>pixelIndices = new ArrayList<Integer>();

	protected boolean isEvalOverridden = false, overrideValue = false;

	public Shape getBoundary() {
		return boundary;
	}
	public DetectionModel getDetectorModel() {
		return model;
	}
	public byte getLatestState() {
		return latestState;
	}
	public void setEvaluatedValue(byte b){
		latestState = b;
	}
	public List<String> getTags() {
		return tags;
	}
	public List<Integer> getPixelIndices() {
		return pixelIndices;
	}

	public synchronized void eval(int pixels[])
	{
		// copy all the pixels that are in my range
		int[] mypixels = new int[pixelIndices.size()];
		int ptr = 0;
		for (Integer i : pixelIndices)
		{
			mypixels[ptr++] = pixels[i];
		}

		// evaluate
		latestState = model.getValue(mypixels);
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer("CanvasDetector[");
		sb.append(boundary).append(',');
		sb.append("tags").append(tags).append(",latestEval=");
		sb.append(latestState).append(',').append(model).append(']');
		return sb.toString();
	}
}