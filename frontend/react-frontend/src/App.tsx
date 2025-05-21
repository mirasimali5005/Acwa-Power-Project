import React, { useEffect, useState } from "react";

interface Employee {
  EMPLOYEE_ID: string;
  FULL_NAME: string;
  EMAIL: string;
  JOB_TITLE: string;
  DEPARTMENT: string;
  HIRE_DATE: string;
  EMPLOYMENT_TYPE: string;
  LOCATION: string;
  MANAGER: string;
}

function App() {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [filterName, setFilterName] = useState("");
  const [filterDept, setFilterDept] = useState("");
  const [selected, setSelected] = useState<Employee | null>(null);

  const fetchEmployees = (name: string, dept: string) => {
    const filters: any = {};
    if (name.trim()) filters.FULL_NAME = name.trim();
    if (dept.trim()) filters.DEPARTMENT = dept.trim();

    fetch("http://127.0.0.1:5000/emp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(filters)
    })
      .then((res) => res.json())
      .then((data) => {
        setEmployees(data);
        setSelected(null);
      })
      .catch((err) => console.error("Failed to fetch employees", err));
  };

  useEffect(() => {
    fetchEmployees("", "");
  }, []);

  const handlePrint = () => {
    if (!selected) return alert("Select an employee to print.");
    const win = window.open();
    if (win) {
      win.document.write("<html><head><title>Print Employee</title></head><body>");
      win.document.write("<h2>Employee Details</h2><ul>");
      for (const [key, value] of Object.entries(selected)) {
        win.document.write(`<li><strong>${key}</strong>: ${value}</li>`);
      }
      win.document.write("</ul></body></html>");
      win.document.close();
    }
  };

  return (
    <div style={{ padding: "1rem", overflowX: "auto" }}>
      <h1>Employee Directory</h1>

      <div style={{ marginBottom: "1rem", display: "flex", gap: "10px", alignItems: "center" }}>
        <input
          type="text"
          placeholder="Filter by full name"
          value={filterName}
          onChange={(e) => setFilterName(e.target.value)}
        />
        <input
          type="text"
          placeholder="Filter by department"
          value={filterDept}
          onChange={(e) => setFilterDept(e.target.value)}
        />
        <button onClick={() => fetchEmployees(filterName, filterDept)}>Apply Filter</button>
        <button onClick={() => {
          setFilterName(""); setFilterDept("");
          fetchEmployees("", "");
        }}>Clear Filters</button>
        <button onClick={handlePrint}>Print</button>
      </div>

      <table border={1} cellPadding={8} style={{ borderCollapse: "collapse", minWidth: "100%" }}>
        <thead>
          <tr>
            <th>EMPLOYEE_ID</th>
            <th>FULL_NAME</th>
            <th>EMAIL</th>
            <th>JOB_TITLE</th>
            <th>DEPARTMENT</th>
            <th>HIRE_DATE</th>
            <th>EMPLOYMENT_TYPE</th>
            <th>LOCATION</th>
            <th>MANAGER</th>
          </tr>
        </thead>
        <tbody>
          {employees.map((emp) => (
            <tr
              key={emp.EMPLOYEE_ID}
              onClick={() => setSelected(emp)}
              style={{
                backgroundColor: selected?.EMPLOYEE_ID === emp.EMPLOYEE_ID ? "#e6f7ff" : "white",
                cursor: "pointer"
              }}
            >
              <td>{emp.EMPLOYEE_ID}</td>
              <td>{emp.FULL_NAME}</td>
              <td>{emp.EMAIL}</td>
              <td>{emp.JOB_TITLE}</td>
              <td>{emp.DEPARTMENT}</td>
              <td>{emp.HIRE_DATE}</td>
              <td>{emp.EMPLOYMENT_TYPE}</td>
              <td>{emp.LOCATION}</td>
              <td>{emp.MANAGER}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
