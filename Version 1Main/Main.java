package Proyecto_Preguntas;

public class Main 
{
	public static void main(String[] args) 
    {
		System.out.println("--------------------");
		System.out.println("INICIO PREPROCESADO");
		System.out.println("-------------------- \n");
		
		// RUTA
		PreProcesado preprocesado = new PreProcesado( "PRUEBAS" );
		
		// FICHERO
		preprocesado.getRawARFF( "train" );
		preprocesado.getRawARFF( "test_unk" );
		
		// FICHERO , BoW-TFIDF , Sparse-NonSparse
		preprocesado.getWordVector( "train" , "BoW"   , "Sparse" );
		preprocesado.getWordVector( "train" , "TFIDF" , "Sparse" );
		preprocesado.getWordVector( "train" , "BoW"   , "NonSparse" );
		preprocesado.getWordVector( "train" , "TFIDF" , "NonSparse" );
		
		// FICHERO , BoW-TFIDF , Sparse-NonSparse
		preprocesado.getWordVector( "test_unk" , "BoW"   , "Sparse" );
		preprocesado.getWordVector( "test_unk" , "TFIDF" , "Sparse" );
		preprocesado.getWordVector( "test_unk" , "BoW"   , "NonSparse" );
		preprocesado.getWordVector( "test_unk" , "TFIDF" , "NonSparse" );
		
		// BoW-TFIDF , Sparse-NonSparse
		preprocesado.filtradoAtributos("BoW"   , "Sparse");
		preprocesado.filtradoAtributos("TFIDF" , "Sparse");
		preprocesado.filtradoAtributos("BoW"   , "NonSparse");
		preprocesado.filtradoAtributos("TFIDF" , "NonSparse");
		
		// BoW-TFIDF , Sparse-NonSparse
		//preprocesado.filtradoBarridoAtributos("BoW"   , "Sparse");
		//preprocesado.filtradoBarridoAtributos("TFIDF" , "Sparse");
		//preprocesado.filtradoBarridoAtributos("BoW"   , "NonSparse");
		//preprocesado.filtradoBarridoAtributos("TFIDF" , "NonSparse");
		
		// BoW-TFIDF , Sparse-NonSparse
		preprocesado.RemoveTest("BoW"   , "Sparse");
		preprocesado.RemoveTest("TFIDF" , "Sparse");
		preprocesado.RemoveTest("BoW"   , "NonSparse");
		preprocesado.RemoveTest("TFIDF" , "NonSparse");
		
		// BoW-TFIDF , Sparse-NonSparse
		//preprocesado.compatibleTrain("BoW"   , "Sparse");
		//preprocesado.compatibleTrain("TFIDF" , "Sparse");
		//preprocesado.compatibleTrain("BoW"   , "NonSparse");
		//preprocesado.compatibleTrain("TFIDF" , "NonSparse");
		
		System.out.println("\n--------------------");
		System.out.println("FIN PREPROCESADO");
		System.out.println("-------------------- \n");
    }
}
