package net.electroland.gotham.processing.metaballs;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.collections15.buffer.BoundedFifoBuffer;

import processing.core.PVector;

public class GridHistory implements Iterable<List<PVector>> {

    private BoundedFifoBuffer<List<PVector>> data;
    private List<PVector> latest;

    public GridHistory(int maxLength){
        setMaxLength(maxLength);
    }

    public void setMaxLength(int maxLength) {
        synchronized(this){
            data = new BoundedFifoBuffer<List<PVector>>(maxLength);
        }
    }

    public void addData(List<PVector> repellingPoints){
        synchronized(this){
            latest = repellingPoints;
            if (data.isFull()){
                data.remove();
            }
            data.add(repellingPoints);
        }
    }

    public List<PVector> latest(){
        return latest == null ? Collections.<PVector>emptyList() : latest;
    }

    @Override
    public Iterator<List<PVector>> iterator() {
        return data.iterator();
    }
}