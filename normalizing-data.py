import json
import re

def process_sentence(sentence: str) -> str:
    # turn all .. and ... characters into .
    for span in re.finditer(r"\.\.+\s*[A-ZĐ]", sentence):
        span = span.group()
        processed_span = re.sub(r"\.\.+", ".", span)
        sentence = re.sub(span, processed_span, sentence)
    for span in re.finditer(r"\.\.+\s*[a-zđ]", sentence):
        span = span.group()
        processed_span = re.sub(r"\.\.+", " ", span)
        sentence = re.sub(span, processed_span, sentence)
    # if there are ., characters, turn them into ,
    sentence = re.sub(r"\.+\s*,", ",", sentence)
    # if there are ,. characters, turn them into .
    if re.search(r",\s*\.+\s*[a-z]", sentence):
        sentence = re.sub(r",\s*\.+", ",", sentence)
    sentence = re.sub(r",\s*\.+", ".", sentence)

    sentence = " ".join(sentence.split())

    return sentence

sentence = "Triển lãm có sự tham gia của đại diện 14 trường đại học Canada, bao gồm: Algoma, Prince Edward Island, Thompson Rivers, Manitoba, Kwantlen Polytechnic, Fraser Valley, Vancouver Island, Crandall, Ontario Tech, Regina, Memorial - Newfoundland, Brock, Royal Roads và Queen.\n\nĐại diện ban tổ chức cho biết, Canada có nền giáo dục phát triển, đảm bảo cho sinh viên quốc tế tiếp cận nền kiến thức và trải nghiệm chuẩn mực. Điều này được chứng minh qua con số 621.656 sinh viên quốc tế học tập tại nước này, theo báo cáo của Statista năm 2021.\n\nĐơn vị này cũng chọn các trường tham gia sự kiện theo tiêu chí chất lượng đào tạo, chính sách học bổng và môi trường học tập, sinh hoạt cho sinh viên. Trong đó có nhiều trường đạt thứ hạng cao trong bảng xếp hạng QS như Đại học Brock, Queen hay Đại học Memorial của Newfoundland.\n\nVị đại diện cũng khẳng định hiện tại là thời điểm thích hợp để du học Canada. Mức học phí học tại đây trong năm học 2022-2023 cho bậc cao đẳng dao động từ 12.000 đến 15.000 CAD mỗi năm; cử nhân là 17.000 - 38.000 CAD một năm và 19.000 - 32.000 CAD đối với bậc thạc sĩ.\n\nBên cạnh đó, những năm gần đây, Chính phủ Canada đã đưa ra nhiều chính sách đẩy mạnh hỗ trợ du học sinh. Ví dụ như cấp giấy phép lao động (work permit) lên đến ba năm cho sinh viên sau khi tốt nghiệp.\n\nSinh viên quốc tế có thể tối ưu hóa giờ làm thêm phù hợp với thời gian biểu theo chính sách bãi bỏ quy định chỉ được làm tối đa 20 giờ mỗi tuần. Do đó, du học sinh có thể làm thêm không giới hạn về thời gian.\n\nĐồng thời, sinh viên có thể tiết kiệm chi phí sinh hoạt bằng cách ở trong khu viên trường cung cấp với mức phí 12.000 - 15.000 CAD mỗi năm. Chi phí thay đổi phụ thuộc vào số lượng người ở và dịch vụ ăn uống. Du học sinh cũng có thể ở ngoài khuôn viên, thuê căn hộ hai phòng ngủ với mắc giá trung bình là 1.500 CAD mỗi tháng hay sống cùng người bản xứ (homestay) với chi phí từ 400 đến 800 CAD mỗi tháng.\n\nNgoài ra, phương tiện duy chuyển tại Canada rất đa dạng. Sinh viên có thể lựa chọn phương thức thuận tiện nhất hoặc đi học, đạp xe đến trường. Thông thường, chi phí di chuyển bằng phương tiện giao thông công cộng: xe buýt, tàu điện ngầm, tàu hỏa... dao động khoảng 150 CAD một tháng.\n\nThiên Minh"

print(process_sentence(sentence))

# data = json.load(open("annotations/ise-dsc01-train.json"))
# for id in data:
#     item = data[id]
#     if item["evidence"] is None:
#         continue
#     item["context"] = process_sentence(item["context"])
#     item["evidence"] = process_sentence(item["evidence"])
#     data[id] = item

# json.dump(data, open("annotations/ise-dsc01-train.json", "w+"), ensure_ascii=False, indent=4)