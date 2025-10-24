import openpyxl


workbook = openpyxl.load_workbook("ISSB26.xlsx")
sheet = workbook.active

with open("ISSB26.fasta", "w") as file:
    # 初始化计数器
    count = 0


    for row in sheet.iter_rows(min_row=2, values_only=True):


        mutation = row[0]
        sequence = row[1]

        fasta_string = f"{mutation}\n{sequence}\n"

        file.write(fasta_string)

        count += 1
