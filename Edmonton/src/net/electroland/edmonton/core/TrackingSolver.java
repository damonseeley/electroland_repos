package net.electroland.edmonton.core;

public class TrackingSolver {

	/*
	// sort sensors and tracks spatially RtoL or LtoR before calling this.

	public Solution solve(Solution bestSolutionSoFar, Solution partialSolution, Sensors remainingSensors, Tracks remainingTracks) {
		if partialSolution.getsScore() >= bestSolutionSoFar.getScore() {
			return bestSolutionSoFar;
		} else if (remainingSensors == null) {
			// all done
			add penalty for unmatched tracks to bestSolutionSoFar's score
			return bestSolutionSoFar;
		} else {
			Sensor s = remainingSensors.pop(); // remove and return first element
			for(Track t : tracks) {
				Tracks tracksWithoutT = remainingTracks.remove(t); // use a non destructive remove, should create a new list
				//optionally disallow s and t as a match because they are too far apart
				Solution newPartial = partialSolution.addMatch(new Match(s,t)); // non destructive, should create a new solution
				Solution newSolution = solve(bestSolutionSoFar, newPartial, remainingSensors, tracksWithoutT)
						if(newSolution.getScore() < bestSolutionSoFar.getScore()) {
							bestSolutionSoFar = newSolution;
						}
			}
			// see if we are better off with creating a new match
			Solution newPartial = partialSolution.addMatch(new Match(s, new TrackAtLocationOfS());
			add new track penalty to newPartial's score
			Solution newSolution = solve(bestSolutionSoFar, newPartial, remainingSensors, remainingTracks)
			if(newSolution.getScore() < bestSolutionSoFar.getScore()) {
				bestSolutionSoFar = newSolution;
			}
			return bestSolutionSoFar;

		}

	}
	 */



}
