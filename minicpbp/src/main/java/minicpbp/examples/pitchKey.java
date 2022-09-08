package minicpbp.examples;

import minicpbp.cp.Factory;
import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Scanner;

import static minicpbp.cp.Factory.makeIntVar;

public class pitchKey {
    public static final int nbVar = 128;
    public static final int nbVarPerBar = 16;
    public static final int nbBar = nbVar / nbVarPerBar;
    public static final int nbVal = 50;
    public static final int[] CMajorPitchClass = {0, 2, 4, 5, 7, 9, 11};
    public static final int nbPitchClass = 12;
    public static final int rhythmOnsetToken = 2;
    public static final int rhythmHoldToken = 1;
    public static final int rhythmRestToken = 0;
    public static final int pitchHoldToken = 48;
    public static final int pitchRestToken = 49;

    public static void main(String[] args) {
        String filename = args[0];
        int nbSample = Integer.parseInt(args[1]);
        int idx = Integer.parseInt(args[2]);
        String filenameTokenRhythm = args[3];
        double oracleWeight = Double.parseDouble(args[4]);
        int K = Integer.parseInt(args[5]);
        int groupSize = Integer.parseInt(args[6]);

        int currentSample = -1;

        int pitchOccInRange = Math.floorDiv(nbVal - 2, nbPitchClass);
        int[][] pitchIdx = new int[nbPitchClass][pitchOccInRange];
        for (int i = 0; i < nbPitchClass; i++) {
            for (int j = 0; j < pitchOccInRange; j++) {
                pitchIdx[i][j] = (j * nbPitchClass) + i;
            }
        }

        try {
            Scanner scannerFilename = new Scanner(new FileReader("minicpbp/examples/data/MusicCP/" + filename));
            Scanner scannerTokenRhythm = new Scanner(new FileReader("minicpbp/examples/data/MusicCP/" + filenameTokenRhythm));
            redirectStdout(filename);

            for (int j = 0; j < nbSample; j++) {
                currentSample = j;

                Solver cp = Factory.makeSolver();

                IntVar[] x = new IntVar[nbVar];
                for (int i = 0; i < nbVar; i++) {
                    x[i] = makeIntVar(cp, 0, nbVal-1);
                    x[i].setName("x"+"["+i+"]");
                }
                int onsetCount = initVar(x, idx, groupSize, scannerTokenRhythm, scannerFilename);

                double[] marginal = new double[nbVal];
                int[] v = new int[nbVal];
                for (int i = 0; i < nbVal; i++) {
                    marginal[i] = (x[idx].contains(i) ? x[idx].marginal(i) : 0);
                    v[i] = i;
                }

                IntVar[] x_subset = Arrays.copyOfRange(x, 0, groupSize * nbVarPerBar);
                IntVar[] o = new IntVar[nbPitchClass];
                for (int i = 0; i < nbPitchClass; i++) {
                    boolean inKey = false;
                    for (int k = 0; k < CMajorPitchClass.length; k++) {
                        if (CMajorPitchClass[k] == i) {
                            inKey = true;
                            break;
                        }
                    }
                    if (inKey) {
                        o[i] = makeIntVar(cp, K, onsetCount);
                    }
                    else {
                        o[i] = makeIntVar(cp, 0, onsetCount);
                    }
                    o[i].setName("o"+"["+i+"]");
                }
                for (int i = 0; i < nbPitchClass; i++) {
                    cp.post(Factory.among(x_subset, pitchIdx[i], o[i]));
                }

                IntVar s = makeIntVar(cp, onsetCount, onsetCount);
                cp.post(Factory.lessOrEqual(Factory.sum(o), s));

                if (idx >= (groupSize * nbVarPerBar)) {
                    oracleWeight = 1.0;
                }
                Constraint orac = Factory.oracle(x[idx], v, marginal);
                orac.setWeight(oracleWeight);
                cp.post(orac);

                cp.fixPoint();
                cp.beliefPropa();

                for (int i = 0; i < nbVal; i++) {
                    System.out.print((x[idx].contains(i) ? x[idx].marginal(i) : 0) + " ");
                }
                System.out.println();
            }
            scannerFilename.close();
            scannerTokenRhythm.close();
        }
        catch (IOException e) {
            System.err.println("Error 1 (" + currentSample + "): " + e) ;
            System.exit(1) ;
        }
        catch (InconsistencyException e) {
            System.err.println("Error 2 (" + currentSample + "): Inconsistency Exception");
            System.exit(2) ;
        }
        catch (Exception  e) {
            System.err.println("Error 3: (" + currentSample + "): " + e);
            System.exit(3) ;
        }
    }

    public static void redirectStdout(String filename) {
        try {
            PrintStream fileOut = new PrintStream("minicpbp/examples/data/MusicCP/" + filename.substring(0, filename.length() - 4) + "_results.dat");
            System.setOut(fileOut);
        }
        catch(FileNotFoundException e) {
            System.err.println("Error 1: " + e.getMessage()) ;
            System.exit(1) ;
        }
    }

    public static int initVar(IntVar[] x, int idx, int groupSize, Scanner sRhythm, Scanner sPitch) {
        int onsetCount = 0;
        int maxIdx = groupSize * nbVarPerBar;

        // read rhythm tokens until current idx
        for (int i = 0; i < idx; i++) {
            // count nb of onset rhythm token
            int tokenRhythm = sRhythm.nextInt();
            if (tokenRhythm == rhythmOnsetToken && i < maxIdx) {
                onsetCount++;
            }

            // set value of previously fixed pitch vars
            int tokenPitch = sPitch.nextInt();
            x[i].assign(tokenPitch);
        }

        // read current rhythm token
        int token = sRhythm.nextInt();
        if (token == rhythmOnsetToken) {
            x[idx].remove(pitchHoldToken);
            x[idx].remove(pitchRestToken);
            if (idx < maxIdx) {
                onsetCount++;
            }
        }
        else {
            x[idx].assign(token == rhythmHoldToken ? pitchHoldToken : pitchRestToken);
        }

        // set marginals for current variable
        for (int i = 0; i < nbVal; i++) {
            double score = sPitch.nextDouble();
            if (x[idx].contains(i)) {
                x[idx].setMarginal(i, score);
            }
        }
        x[idx].normalizeMarginals();

        // read the later rhythm tokens to keep counting nb of onset and
        // set uniform marginals for following pitch variables
        for (int i = idx + 1; i < nbVar; i++) {
            int tokenRhythm = sRhythm.nextInt();
            if (tokenRhythm == rhythmHoldToken) {
                x[i].assign(pitchHoldToken);
            }
            else if (tokenRhythm == rhythmRestToken) {
                x[i].assign(pitchRestToken);
            }
            else {
                x[i].remove(pitchHoldToken);
                x[i].remove(pitchRestToken);
                if (i < maxIdx) {
                    onsetCount++;
                }
            }

            x[i].resetMarginals();
            x[i].normalizeMarginals();
        }

        return onsetCount;
    }
}

