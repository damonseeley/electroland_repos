package net.electroland.blobDetection.match;

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeSet;
import java.util.Vector;

import net.electroland.blobDetection.Blob;
import net.electroland.blobDetection.Region;

public class CSP {
	boolean stopProcessing = false;
	boolean firstSolutionFound = false;

	public static final Blob UNMATCHED = new Blob(-1,-1);
//	float nonMatchPenalty;
//	float provisionalPentaly;

//	HashMap<Track, HashSet<Blob>> possibleTracks = new HashMap<Track, HashSet<Blob>>();
	HashMap<Track, TreeSet<Blob>> possibleTracks = new HashMap<Track, TreeSet<Blob>>();
	HashMap<Blob, HashSet<Track>> possibleBlobs = new HashMap<Blob, HashSet<Track>>();
	Region region;

	public CSP(Region r) {
		region = r;
	}
	public void stopProcessing() {
		stopProcessing = true;	
	}

	public Solution solve(Grid g, Vector<Track> tracks) {
		return solve(g, tracks, false);
	}

	public Solution solve(Grid g, Vector<Track> tracks, boolean singlePass) {
		stopProcessing = singlePass;
		possibleTracks.clear();
		possibleBlobs.clear();

		for(Track t : tracks) {
			HashSet<Blob> blobs = g.getPossibleMatchs(t.x, t.y);
			for(Blob b : blobs) {
				HashSet<Track> pTracks= possibleBlobs.get(b);
				if(pTracks == null) {
					pTracks = new HashSet<Track>();
					possibleBlobs.put(b, pTracks);
				} 
				pTracks.add(t);				
			}

			TreeSet<Blob> sortedBlobs = new TreeSet<Blob>(new CompareByDist(t));
			sortedBlobs.addAll(blobs);

			//	System.out.println("----");
			//	for(Blob b: sortedBlobs) {

			//		System.out.println("Dist = " + b.dist(t.x, t.y));

			//	}


			possibleTracks.put(t,sortedBlobs);
		}

		Solution s = solve(new Solution(possibleTracks, possibleBlobs), Float.MAX_VALUE);
		s.unmachedBlobs = (HashSet<Blob>) g.allBlobs.clone();
		s.unmachedBlobs.removeAll(s.assigned.values());


		return s;

	}

	public class Solution {
		HashMap<Track, Blob> assigned =  new HashMap<Track, Blob>();
		HashMap<Track, TreeSet<Blob>> tracks = new HashMap<Track, TreeSet<Blob>> ();
		HashMap<Blob, HashSet<Track>> blobs = new HashMap<Blob, HashSet<Track>>();
		HashSet<Blob> unmachedBlobs;
		float score;

		public String toString() {
			StringBuffer sb = new StringBuffer();
			sb.append("Solution -- \n");

			sb.append("  Tracks\n");
			for(Map.Entry<Track, TreeSet<Blob>> e : tracks.entrySet()) {
				sb.append("    " + e + "\n");
			}
			sb.append("  Blobs\n");
			for(Map.Entry<Blob, HashSet<Track>> e : blobs.entrySet()) {
				sb.append("    " + e + "\n");
			}
			if(unmachedBlobs != null) {

				sb.append("  freeBlobs\n") ;
				for(Blob b :unmachedBlobs) {
					sb.append("    " + b + "\n");
				}
			}
			sb.append("  Assigned\n");
			for(Map.Entry<Track, Blob> e : assigned.entrySet()) {
				sb.append("    " + e + "\n");
			}

			sb.append("  Score=" + score+ "\n");
			return sb.toString();

		}


		public Solution(HashMap<Track, TreeSet<Blob>> ts, HashMap<Blob, HashSet<Track>> bs) {
			for(Map.Entry<Track, TreeSet<Blob>> te : ts.entrySet()) {
				tracks.put(te.getKey(), (TreeSet<Blob>)te.getValue().clone());
			}

			for(Map.Entry<Blob, HashSet<Track>> be : bs.entrySet()) {
				blobs.put(be.getKey(), (HashSet<Track>)be.getValue().clone());
			}			
			score = 0;


		}

		public Solution(Solution s) {
			assigned = (HashMap<Track, Blob>)s.assigned.clone();

			for(Map.Entry<Track, TreeSet<Blob>> te : s.tracks.entrySet()) {
				tracks.put(te.getKey(), (TreeSet<Blob>)te.getValue().clone());
			}

			for(Map.Entry<Blob, HashSet<Track>> be : s.blobs.entrySet()) {
				blobs.put(be.getKey(), (HashSet<Track>)be.getValue().clone());
			}

			score = s.score;

		}

		public void addMatch(Track track, Blob blob) {
//			System.out.println("addMatch-" + track + " " + blob);
			if(blob == UNMATCHED) {
				tracks.remove(track);
				assigned.put(track, UNMATCHED);
				score += region.nonMatchPenalty;
			} else {
				HashSet<Track> tracksToUpdate = blobs.get(blob);
				for(Track t : tracksToUpdate) {
					TreeSet<Blob> blobsToRemove = tracks.get(t);
					if(blobsToRemove != null) { // may have been assigned and already removed
						tracks.get(t).remove(blob);
					}
				}
				blobs.remove(blob);
				tracks.remove(track);
				assigned.put(track, blob);
				score += blob.dist(track.x, track.y);

				if(track.isProvisional) {
					score += region.provisionalPentaly;
				}
			}
		}
	}


	public Solution solve(Solution solution, float oldBestScore) {
//		System.out.println(":Solve - alpha " +  oldBestScore);
//		System.out.println(solution);


		if(stopProcessing) {
			if(oldBestScore <  Float.MAX_VALUE) { // if found a solution already
				return solution;
			}
		}

		// check end conditions
		if(oldBestScore < solution.score) {
//			System.out.println("buble up null");
			return null; // prune
		}


		boolean isDone = false;

		if(solution.tracks.isEmpty()) { // add up score for unmachted blobs
			// only penalize for unmatched tracks
			//		System.out.println("## no tracks");
//			solution.score += solution.blobs.size() * nonMatchPenalty;
			isDone = true;
			firstSolutionFound = true;
		}

		if(solution.blobs.isEmpty()) { // add up score for unmatched tracks
//			System.out.println("adding pentalty for extra tracks: " + solution.tracks.size());			
			//		System.out.println("## no blobs");
			solution.score += solution.tracks.size() * region.nonMatchPenalty;
			isDone = true;
			firstSolutionFound = true;
		}


		if(isDone) {
//			System.out.println("bubleup solution" + solution);
			if(oldBestScore < solution.score) {
				//			System.out.println("@@ pruning returning null");
				return null; // prune
			} else {
//				System.out.println("++ buble up");
//				System.out.println(solution);
//				System.out.println("++");
				return solution;
			}
		}





		// get most constrained
		Track mostConstrained = null;
		TreeSet<Blob> mcBlobs = null;
		int constraintSize = Integer.MAX_VALUE;
		for(Map.Entry<Track, TreeSet<Blob>> entry : solution.tracks.entrySet()) {
			if(entry.getValue().size() < constraintSize) {
				mostConstrained = entry.getKey();
				mcBlobs = entry.getValue();
				constraintSize = mcBlobs.size();
			}
			if(constraintSize == 0) break; // can't be less than 0 so don't check rest
		}

		if(constraintSize == 0) { //leaf
			Solution newSolution = new Solution(solution);
			newSolution.addMatch(mostConstrained, UNMATCHED);
			Solution possibleSolution = solve(newSolution,oldBestScore);
			//		System.out.println("## returning leaf (no chioce)");
			if(possibleSolution != null) {
				if(possibleSolution.score < oldBestScore) {
					return possibleSolution;
				} else {
					return null;
				}
			}
		} 

		Solution bestSolution = solution;
		float bestScore = oldBestScore;

		//now sorted once during initial solution creation
//		TreeSet<Blob> sortedBlobs = new TreeSet<Blob>(new CompareByDist(mostConstrained));
		//	sortedBlobs.addAll(mcBlobs);
		// sorts blobs in closest to farthest so does a greedy search first
		// should result in better heuristic pruning

		boolean newReturn = false;
		for(Blob b : mcBlobs) {
			Solution newSolution = new Solution(solution);
			newSolution.addMatch(mostConstrained, b);
			Solution possibleSolution = solve(newSolution,bestScore);
			if(possibleSolution != null) {
				if(possibleSolution.score < bestScore) {
					bestSolution = possibleSolution;
					bestScore = bestSolution.score;
					newReturn = true;
					//		System.out.println("updateing bestscore to " + bestScore);
				}
			}
			if(firstSolutionFound && stopProcessing) {
				return bestSolution;
			}

		}

		Solution newSolution = new Solution(solution);
		newSolution.addMatch(mostConstrained, UNMATCHED);
		Solution possibleSolution = solve(newSolution,bestScore);
		if(possibleSolution != null) {
			if(possibleSolution.score < bestScore) {
				newReturn = true;
				bestSolution = possibleSolution;
				bestScore = bestSolution.score;
			}
		}

		//	System.out.println("## return best");
		//	System.out.println("++");
		//	System.out.println(bestSolution);
		//	System.out.println("++");
		if(newReturn) {
			return bestSolution;
		} else {
			return null;
		}






	}


	public class CompareByDist implements Comparator<Blob> {
		float x;
		float y;
		public CompareByDist(Track t) {
			x = t.x;
			y = t.y;
		}

		public int compare(Blob o1, Blob o2) {
			return (int) (o1.distSqr(x, y) - o2.distSqr(x, y));
		}
	}




	public static class TestStopper extends Thread {
		CSP csp;

		public TestStopper(CSP csp) {
			this.csp = csp;
		}
		public void run() {
			try {
				Thread.sleep(2000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("stoping");
			csp.stopProcessing();
			System.out.println("stopped");
		}
	}

}
