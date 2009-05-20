package net.electroland.elvisVideoProcessor.util;

import java.util.concurrent.ArrayBlockingQueue;

public class SwapBuffer<T> {
	ArrayBlockingQueue<T> toProcess = new ArrayBlockingQueue<T>(2);
	ArrayBlockingQueue<T> processed = new ArrayBlockingQueue<T>(2);
	T lastProcessed = null;
	
	public T takeToProcess() throws InterruptedException {
		return toProcess.take();
	}

	
	public SwapBuffer(T initToProc1, T initToProc2) {
			toProcess.add(initToProc1);
			toProcess.add(initToProc2);
	}


	public void putProcessed(T t) throws InterruptedException {
		processed.put(t);
	}

	/**
	 * returns ready element T.  Assumes that last returned element is now free to reuse.
	 * @return
	 * @throws InterruptedException
	 */	
	public T takeProcessed() throws InterruptedException {
		T free = lastProcessed;
		lastProcessed = processed.take();
		if(free != null) {
			toProcess.put(free);
		}
		return lastProcessed;
	}

}
