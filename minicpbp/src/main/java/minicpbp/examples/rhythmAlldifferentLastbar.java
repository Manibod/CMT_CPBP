package minicpbp.examples;

import minicpbp.cp.Factory;
import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Scanner;

import static minicpbp.cp.Factory.*;

public class rhythmAlldifferentLastbar {

    public static int nbVar = 128;
    public static int nbVarPerBar = 16;
    public static int nbBar = nbVar / nbVarPerBar;
    public static int nbVal = 3;
    public static int onSetToken = 2;

    public static void main(String[] args) {
        String filename = args[0];
        int nbSample = Integer.parseInt(args[1]);
        int idx = Integer.parseInt(args[2]);
        double oracleWeight = Double.parseDouble(args[3]);
        int groupSize = Integer.parseInt(args[4]);

        int currentSample = -1;
        int inconsistencyCount = 0;

        try {
            Scanner scanner = new Scanner(new FileReader("minicpbp/examples/data/MusicCP/" + filename));
            redirectStdout(filename);

            for (int j = 0; j < nbSample; j++) {
                try {
                    currentSample = j;

                    Solver cp = Factory.makeSolver();

                    IntVar[] x = new IntVar[nbVar];
                    for (int i = 0; i < nbVar; i++) {
                        x[i] = makeIntVar(cp, 0, nbVal - 1);
                        x[i].setName("x" + "[" + i + "]");
                    }
                    initVar(x, nbVar, nbVal, idx, scanner);

                    double[] marginal = new double[nbVal];
                    int[] v = new int[nbVal];
                    for (int i = 0; i < nbVal; i++) {
                        marginal[i] = (x[idx].contains(i) ? x[idx].marginal(i) : 0);
                        v[i] = i;
                    }

                    IntVar[] o = new IntVar[nbBar];
                    for (int i = 0; i < nbBar; i++) {
                        o[i] = makeIntVar(cp, 0, nbVarPerBar);
                        o[i].setName("o" + "[" + i + "]");
                    }

                    for (int i = 0; i < nbBar; i++) {
                        IntVar[] x_subset = Arrays.copyOfRange(x, i * nbVarPerBar, (i + 1) * nbVarPerBar);
                        cp.post(Factory.among(x_subset, onSetToken, o[i]));
                    }

                    for (int i = 0; i < nbBar; i += groupSize) {
                        int k = i + groupSize > nbBar ? nbBar : i + groupSize;
                        IntVar[] o_subset = Arrays.copyOfRange(o, i, k);
                        cp.post(Factory.allDifferentAC(o_subset));
                        if (groupSize != 1) {
                            cp.post(Factory.lessOrEqual(mul(o[k - 2], 2), o[k - 1]));
                            cp.post(Factory.lessOrEqual(o[k - 1], mul(o[k - 2], 2)));
                        }
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
                catch (InconsistencyException e) {
                    inconsistencyCount++;
                    for (int i = 0; i < nbVal; i++) {
                        System.out.print((i == 0 ? 1 : 0) + " ");
                    }
                    System.out.println();
                }
            }
            scanner.close();

            FileWriter myWriter = new FileWriter("minicpbp/examples/data/MusicCP/nb_inconsistency_error.txt", true);
            myWriter.write(inconsistencyCount + "\n");
            myWriter.close();
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

    public static void initVar(IntVar[] x, int nbVar, int nbVal, int idx, Scanner s){
        // set value of previously fixed vars
        for (int i = 0; i < idx; i++) {
            int token = s.nextInt();
            x[i].assign(token);
        }

        // set marginals for variable being fixed
        for (int i = 0; i < nbVal; i++) {
            double score = s.nextDouble();
            if (x[idx].contains(i)) {
                x[idx].setMarginal(i, score);
            }
        }
        x[idx].normalizeMarginals();

        // set uniform marginals for following variables
        for (int i = idx + 1; i < nbVar; i++) {
            x[i].resetMarginals();
            x[i].normalizeMarginals();
        }
    }
}
