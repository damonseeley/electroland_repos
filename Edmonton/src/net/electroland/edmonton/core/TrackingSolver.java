package net.electroland.edmonton.core;

public class TrackingSolver {
	
	/*
	sort sensors and track spatially rtol or ltor before calling this.
	
	public Solution solve(Sensors sensors, Tracks tracks, Solution bestScoreSoFar) {
		if sensors.isEmpty() {
			// for each unamtched track add to solution and update score with some penalty
			return new Solution();
		} else {
			Sensor s = sensors.removeFirst();
			for(Track t:tracks) {
				Match m = new Match(s,t)
				//is match valid? ie too far apart,etc
			Solution partialSolution = solve(sensors, tracks, bestScoreSoFar);
			if(partialSolution != null) {
				paritalSolution.addMatch(m)
				if(paritalSolution.getScore() < bestSoFar.getScore()) {
					bestScoreSoFar =  paritalSolution;
				} else {
					return null;
				}
			}
			// create one more solution with a new track and compare
			return bestScoreSoFar;
			}
			
		}
	}
	*/

}
