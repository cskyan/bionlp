from __future__ import print_function
import sys
import os
import site
import argparse
from io import StringIO
from clinphen_src import get_phenotypes
from clinphen_src import src_dir
# srcDir = src_dir.get_src_dir()
srcDir = os.path.join([s for s in site.getsitepackages() if 'site-packages' in s][0], 'clinphen_src')
myDir = "/".join(os.path.realpath(__file__).split("/")[:-1])

def load_common_phenotypes(commonFile):
  returnSet = set()
  for line in open(commonFile): returnSet.add(line.strip())
  return returnSet

def main(inputFile, custom_thesaurus="", rare=False):
  hpo_main_names = os.path.join(srcDir, "data", "hpo_term_names.txt")

  def getNames():
    returnMap = {}
    for line in open(hpo_main_names):
      lineData = line.strip().split("\t")
      returnMap[lineData[0]] = lineData[1]
    return returnMap
  hpo_to_name = getNames()

  inputStr = ""
  for line in inputFile.readlines() if type(inputFile) is StringIO else open(inputFile): inputStr += line
  if not custom_thesaurus: returnString = get_phenotypes.extract_phenotypes(inputStr, hpo_to_name)
  else: returnString = get_phenotypes.extract_phenotypes_custom_thesaurus(inputStr, custom_thesaurus, hpo_to_name)
  if not rare: return returnString
  items = returnString.split("\n")
  returnList = []
  common = load_common_phenotypes(os.path.join(srcDir, "data", "common_phenotypes.txt"))
  for item in items:
    HPO = item.split("\t")[0]
    if HPO in common: continue
    returnList.append(item)
  return "\n".join(returnList)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('recordfile', type=str, nargs='?', help='The file containing the clinical notes.')
  parser.add_argument('--update', action='store_true', help='Update the HPO thesaurus')
  parser.add_argument('--umls_setup', type=str, help='Set up the UMLS thesaurus by providing the path to the downloaded, unzipped UMLS directory')
  parser.add_argument('--umls', action='store_true', help='Use the UMLS thesaurus')
  parser.add_argument('--custom_thesaurus', help='Use a custom thesaurus whose filepath you specify. The format is: HPO_ID<tab>Synonym')
  parser.add_argument('--rare', action='store_true', default=False, help='Limit to rare phenotypes')
  args = parser.parse_args()
  if args.update:
    os.system("sh " + os.path.join(srcDir, "hpo_setup.sh") + " " + srcDir)
    quit()
  if args.umls_setup:
    os.system("sh " + os.path.join(srcDir, "umls_thesaurus_extraction.sh")+ " " + args.umls_setup + " " + os.path.join(srcDir, "data") + " " + os.path.join(srcDir, "prep_thesaurus.py"))
    quit()
  custom_thesaurus = ""
  if args.umls: custom_thesaurus = os.path.join(srcDir, "data", "hpo_umls_thesaurus.txt")
  if args.custom_thesaurus: custom_thesaurus = args.custom_thesaurus
  print(main(args.recordfile, custom_thesaurus, args.rare))
